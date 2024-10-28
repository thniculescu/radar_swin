# --------------------------------------------------------
# Swin Transformer
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu
# --------------------------------------------------------
# %%
# %load_ext autoreload
# %autoreload 2
import os
import time
import json
import random
import argparse
import datetime
import numpy as np

import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist

from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from timm.utils import accuracy, AverageMeter

from config import get_config
from models.build_model import build_model
from data import build_loader
from lr_scheduler import build_scheduler
from optimizer import build_optimizer
from logger import create_logger
from utils import load_checkpoint, load_pretrained, save_checkpoint, NativeScalerWithGradNormCount, auto_resume_helper, reduce_tensor
from losses import RadarSwinLoss
import torchinfo
from metrics import calc_ap, get_tp_fp_conf
from do_plots import do_plots

# pytorch major version (1.x or 2.x)
PYTORCH_MAJOR_VERSION = int(torch.__version__.split('.')[0])


# def parse_option():
#     parser = argparse.ArgumentParser('Swin Transformer training and evaluation script', add_help=False)
#     parser.add_argument('--cfg', type=str, required=True, metavar="FILE", help='path to config file', )
#     parser.add_argument(
#         "--opts",
#         help="Modify config options by adding 'KEY VALUE' pairs. ",
#         default=None,
#         nargs='+',
#     )

#     # easy config modification
#     parser.add_argument('--batch-size', type=int, help="batch size for single GPU")
#     parser.add_argument('--data-path', type=str, help='path to dataset')
#     parser.add_argument('--zip', action='store_true', help='use zipped dataset instead of folder dataset')
#     parser.add_argument('--cache-mode', type=str, default='part', choices=['no', 'full', 'part'],
#                         help='no: no cache, '
#                              'full: cache all data, '
#                              'part: sharding the dataset into nonoverlapping pieces and only cache one piece')
#     parser.add_argument('--pretrained',
#                         help='pretrained weight from checkpoint, could be imagenet22k pretrained weight')
#     parser.add_argument('--resume', help='resume from checkpoint')
#     parser.add_argument('--accumulation-steps', type=int, help="gradient accumulation steps")
#     parser.add_argument('--use-checkpoint', action='store_true',
#                         help="whether to use gradient checkpointing to save memory")
#     parser.add_argument('--disable_amp', action='store_true', help='Disable pytorch amp')
#     parser.add_argument('--amp-opt-level', type=str, choices=['O0', 'O1', 'O2'],
#                         help='mixed precision opt level, if O0, no amp is used (deprecated!)')
#     parser.add_argument('--output', default='output', type=str, metavar='PATH',
#                         help='root of output folder, the full path is <output>/<model_name>/<tag> (default: output)')
#     parser.add_argument('--tag', help='tag of experiment')
#     parser.add_argument('--eval', action='store_true', help='Perform evaluation only')
#     parser.add_argument('--throughput', action='store_true', help='Test throughput only')

#     # distributed training
#     # for pytorch >= 2.0, use `os.environ['LOCAL_RANK']` instead
#     # (see https://pytorch.org/docs/stable/distributed.html#launch-utility)
#     if PYTORCH_MAJOR_VERSION == 1:
#         parser.add_argument("--local_rank", type=int, required=True, help='local rank for DistributedDataParallel')

#     # for acceleration
#     parser.add_argument('--fused_window_process', action='store_true',
#                         help='Fused window shift & window partition, similar for reversed part.')
#     parser.add_argument('--fused_layernorm', action='store_true', help='Use fused layernorm.')
#     ## overwrite optimizer in config (*.yaml) if specified, e.g., fused_adam/fused_lamb
#     parser.add_argument('--optim', type=str,
#                         help='overwrite optimizer if provided, can be adamw/sgd/fused_adam/fused_lamb.')

#     args, unparsed = parser.parse_known_args()

#     config = get_config(args)

#     return args, config

def parse_option():
    # CONFIG_PATH = '/imec/other/dl4ms/nicule52/work/radarswin/radar-swin/configs/radarswin/bbox_vr_6sw_biglast.yaml'
    print("CWDDD:", os.getcwd())

    # CONFIG_PATH = './configs/radarswin/bbox_no_merge_best.yaml'
    # CONFIG_PATH = './configs/radarswin/alpha_small.yaml'
    CONFIG_PATH = './configs/radarswin/alpha_small_static.yaml'

    parser = argparse.ArgumentParser('Swin Transformer training and evaluation script', add_help=False)
    parser.add_argument('--cfg', type=str, required=False, metavar="FILE", help='path to config file', default=CONFIG_PATH)
    parser.add_argument(
            "--opts",
            help="Modify config options by adding 'KEY VALUE' pairs. ",
            default=None,
            nargs='+',
        )

    args, unparsed = parser.parse_known_args()
    config = get_config(args)

    return args, config


def main(config, logger):
    dataset_train, dataset_val, data_loader_train, data_loader_val, mixup_fn = build_loader(config)

    logger.info(f"Creating model:{config.MODEL.TYPE}/{config.MODEL.NAME}")
    model = build_model(config)

    torchinfo.summary(model, input_size=(config.DATA.BATCH_SIZE, config.MODEL.RADARSWIN.IN_CHANS, config.DATA.INPUT_SIZE[0], config.DATA.INPUT_SIZE[1]), depth=4, col_names=["input_size", "output_size", "num_params", "mult_adds"])

    # logger.info(str(model))

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"number of params, M: {n_parameters / 1e6}")
    if hasattr(model, 'flops'):
        flops = model.flops()
        logger.info(f"number of GFLOPs: {flops / 1e9}")

    model.cuda()
    model_without_ddp = model

    optimizer = build_optimizer(config, model)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[config.LOCAL_RANK], broadcast_buffers=False)
    loss_scaler = NativeScalerWithGradNormCount()

    if config.TRAIN.ACCUMULATION_STEPS > 1:
        lr_scheduler = build_scheduler(config, optimizer, len(data_loader_train) // config.TRAIN.ACCUMULATION_STEPS)
    else:
        lr_scheduler = build_scheduler(config, optimizer, len(data_loader_train))

    # if config.AUG.MIXUP > 0.:
    #     # smoothing is handled with mixup label transform
    #     criterion = SoftTargetCrossEntropy()
    # elif config.MODEL.LABEL_SMOOTHING > 0.:
    #     criterion = LabelSmoothingCrossEntropy(smoothing=config.MODEL.LABEL_SMOOTHING)
    # else:

    criterion = RadarSwinLoss(config, range_es=config.MODEL.RADARSWIN.RANGE_LOSS_WEIGHT, bbox_size_ori=config.MODEL.RADARSWIN.BBOX_LOSS_WEIGHT, velocity=config.MODEL.RADARSWIN.VELOCITY_LOSS_WEIGHT) # 1, [2, 2], 2

    max_accuracy = 0.0

    if config.TRAIN.AUTO_RESUME:
        resume_file = auto_resume_helper(config.OUTPUT)
        if resume_file:
            if config.MODEL.RESUME:
                logger.warning(f"auto-resume changing resume file from {config.MODEL.RESUME} to {resume_file}")
            config.defrost()
            config.MODEL.RESUME = resume_file
            config.freeze()
            logger.info(f'auto resuming from {resume_file}')
        else:
            logger.info(f'no checkpoint found in {config.OUTPUT}, ignoring auto resume')

    if config.MODEL.RESUME:
        max_accuracy = load_checkpoint(config, model_without_ddp, optimizer, lr_scheduler, loss_scaler, logger)
        acc1, acc5, loss = validate(config, data_loader_val, model, logger)
        logger.info(f"Accuracy of the network on the {len(dataset_val)} test images: {acc1:.1f}%")
        if config.EVAL_MODE:
            return

    if config.MODEL.PRETRAINED and (not config.MODEL.RESUME):
        # for chk_nr in range(5, 475, 5):
        #     config.defrost()
        #     config.MODEL.PRETRAINED = f'/imec/other/dl4ms/nicule52/work/radarswin/radarswin_checkpoints/g_big400/ckpt_epoch_{chk_nr}.pth'
        #     config.freeze()

        load_pretrained(config, model_without_ddp, logger)
        acc1, acc5, loss = validate(config, data_loader_val, model, logger)
        logger.info(f"Accuracy of the network on the {len(dataset_val)} test images: {acc1:.1f}%")

        if config.EVAL_MODE:
            do_plots(config, data_loader_val, model)
            return

    if config.THROUGHPUT_MODE:
        throughput(data_loader_val, model, logger)
        return

    logger.info("Start training")
    start_time = time.time()
    for epoch in range(config.TRAIN.START_EPOCH, config.TRAIN.EPOCHS):
        data_loader_train.sampler.set_epoch(epoch)

        train_one_epoch(config, model, criterion, data_loader_train, optimizer, epoch, mixup_fn, lr_scheduler,
                        loss_scaler, logger=logger)
        if dist.get_rank() == 0 and (epoch % config.SAVE_FREQ == 0 or epoch == (config.TRAIN.EPOCHS - 1)):
            save_checkpoint(config, epoch, model_without_ddp, max_accuracy, optimizer, lr_scheduler, loss_scaler,
                            logger)

        acc1, acc5, loss = validate(config, data_loader_val, model, logger)
        logger.info(f"Accuracy of the network on the {len(dataset_val)} test images: {acc1:.1f}%")
        max_accuracy = max(max_accuracy, acc1)
        logger.info(f'Max accuracy: {max_accuracy:.2f}%')

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logger.info('Training time {}'.format(total_time_str))


def train_one_epoch(config, model, criterion, data_loader, optimizer, epoch, mixup_fn, lr_scheduler, loss_scaler, logger):
    model.train()
    optimizer.zero_grad()

    num_steps = len(data_loader)
    batch_time = AverageMeter()
    loss_meter = AverageMeter()
    norm_meter = AverageMeter()
    scaler_meter = AverageMeter()

    start = time.time()
    end = time.time()
    for idx, (samples, targets) in enumerate(data_loader):
        samples = samples.cuda(non_blocking=True)
        targets = targets.cuda(non_blocking=True)

        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)

        with torch.amp.autocast("cuda", enabled=config.AMP_ENABLE):
            outputs = model(samples)

        loss = criterion(outputs, targets)
        loss = loss / config.TRAIN.ACCUMULATION_STEPS

        # this attribute is added by timm on one optimizer (adahessian)
        is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
        grad_norm = loss_scaler(loss, optimizer, clip_grad=config.TRAIN.CLIP_GRAD,
                                parameters=model.parameters(), create_graph=is_second_order,
                                update_grad=(idx + 1) % config.TRAIN.ACCUMULATION_STEPS == 0)
        if (idx + 1) % config.TRAIN.ACCUMULATION_STEPS == 0:
            optimizer.zero_grad()
            lr_scheduler.step_update((epoch * num_steps + idx) // config.TRAIN.ACCUMULATION_STEPS)
        loss_scale_value = loss_scaler.state_dict()["scale"]

        torch.cuda.synchronize()

        loss_meter.update(loss.item(), targets.size(0))
        if grad_norm is not None:  # loss_scaler return None if not update
            norm_meter.update(grad_norm)
        scaler_meter.update(loss_scale_value)
        batch_time.update(time.time() - end)
        end = time.time()

        if idx % config.PRINT_FREQ == 0:
            lr = optimizer.param_groups[0]['lr']
            wd = optimizer.param_groups[0]['weight_decay']
            memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
            etas = batch_time.avg * (num_steps - idx)
            logger.info(
                f'Train: [{epoch}/{config.TRAIN.EPOCHS}][{idx}/{num_steps}]\t'
                f'eta {datetime.timedelta(seconds=int(etas))} lr {lr:.6f}\t wd {wd:.4f}\t'
                f'time {batch_time.val:.4f} ({batch_time.avg:.4f})\t'
                f'loss {loss_meter.val:.4f} ({loss_meter.avg:.4f})\t'
                f'grad_norm {norm_meter.val:.4f} ({norm_meter.avg:.4f})\t'
                f'loss_scale {scaler_meter.val:.4f} ({scaler_meter.avg:.4f})\t'
                f'mem {memory_used:.0f}MB')
    epoch_time = time.time() - start
    logger.info(f"EPOCH {epoch} training takes {datetime.timedelta(seconds=int(epoch_time))}")


@torch.no_grad()
def validate(config, data_loader, model, logger):
    criterion = RadarSwinLoss(config, range_es=config.MODEL.RADARSWIN.RANGE_LOSS_WEIGHT, bbox_size_ori=config.MODEL.RADARSWIN.BBOX_LOSS_WEIGHT, velocity=config.MODEL.RADARSWIN.VELOCITY_LOSS_WEIGHT) # 1, [2, 2], 2
    model.eval()

    batch_time = AverageMeter()
    loss_meter = AverageMeter()
    AP_meter = AverageMeter()
    ap_dist_tresh = np.array([0.5, 1.0, 2.0, 4.0])
    # ap_dist_tresh = np.array([1])
    thresh_tp_fp_conf = np.empty((len(ap_dist_tresh), 3, 0))
    total_gt = 0

    end = time.time()
    for idx, (images, target) in enumerate(data_loader):
        images = images.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)

        # compute output
        with torch.amp.autocast("cuda", enabled=config.AMP_ENABLE):
            output = model(images)

        # measure accuracy and record loss
        loss = criterion(output, target)

        #calc AP
        tp_fp_conf, num_gt = get_tp_fp_conf(output, target, ap_dist_tresh)
        total_gt += num_gt
        thresh_tp_fp_conf = np.dstack((thresh_tp_fp_conf, tp_fp_conf))
        # print(thresh_tp_fp_conf.shape)

        loss = reduce_tensor(loss)

        loss_meter.update(loss.item(), target.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if idx % config.PRINT_FREQ == 0:
            memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
            logger.info(
                f'Test: [{idx}/{len(data_loader)}]\t'
                f'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                f'Loss {loss_meter.val:.4f} ({loss_meter.avg:.4f})\t'
                # f'detA@2m {detA2_meter.val:.3f} ({detA2_meter.avg:.3f})\t'
                # f'AP {AP_meter.val:.3f} ({AP_meter.avg:.3f})\t'
                f'Mem {memory_used:.0f}MB')
    
    aps = calc_ap(thresh_tp_fp_conf, total_gt, ap_dist_tresh)
    logger.info(f'AP {np.sum(aps) / len(aps):.3f} ({aps})\t')
    # logger.info(f' * Acc@1 {detA2_meter.avg:.3f} Acc@5 {AP_meter.avg:.3f}')
    # return detA2_meter.avg, AP_meter.avg, loss_meter.avg
    return 44, AP_meter.avg, loss_meter.avg


@torch.no_grad()
def throughput(data_loader, model, logger):
    model.eval()

    for idx, (images, _) in enumerate(data_loader):
        images = images.cuda(non_blocking=True)
        batch_size = images.shape[0]
        for i in range(50):
            model(images)
        torch.cuda.synchronize()
        logger.info(f"throughput averaged with 30 times")
        tic1 = time.time()
        for i in range(30):
            model(images)
        torch.cuda.synchronize()
        tic2 = time.time()
        logger.info(f"batch_size {batch_size} throughput {30 * batch_size / (tic2 - tic1)}")
        return


def ddp_main(rank, world_size, device_ids):
    os.environ['LOCAL_RANK'] = str(device_ids[rank])

    args, config = parse_option()

    print(f"rank {rank} / world_size {world_size} / device_ids {device_ids}")
    if config.AMP_OPT_LEVEL:
        print("[warning] Apex amp has been deprecated, please use pytorch amp instead!")

    #TODO: used in original
    # if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
    #     rank = int(os.environ["RANK"])
    #     world_size = int(os.environ['WORLD_SIZE'])
    #     print(f"RANK and WORLD_SIZE in environ: {rank}/{world_size}")
    # else:
    #     rank = -1
    #     world_size = -1
    #     print("NEMA RANK")
    # config.LOCAL_RANK = os.environ['LOCAL_RANK']
    print(f"LOCAL_RANK: {config.LOCAL_RANK}")

    torch.cuda.set_device(config.LOCAL_RANK)


    #TODO: repair distro training
    torch.distributed.init_process_group(backend='nccl', init_method='env://', world_size=world_size, rank=rank)
    torch.distributed.barrier()
    seed = config.SEED + dist.get_rank() #TODO rpair for distro training
    # seed = config.SEED
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True

    #TODO: repair distro training    
    # # linear scale the learning rate according to total batch size, may not be optimal
    linear_scaled_lr = config.TRAIN.BASE_LR * config.DATA.BATCH_SIZE * dist.get_world_size() / 1
    linear_scaled_warmup_lr = config.TRAIN.WARMUP_LR * config.DATA.BATCH_SIZE * dist.get_world_size() / 1
    linear_scaled_min_lr = config.TRAIN.MIN_LR * config.DATA.BATCH_SIZE * dist.get_world_size() / 1
    
    # linear_scaled_lr = config.TRAIN.BASE_LR * config.DATA.BATCH_SIZE * dist.get_world_size() / 512.0
    # linear_scaled_warmup_lr = config.TRAIN.WARMUP_LR * config.DATA.BATCH_SIZE * dist.get_world_size() / 512.0
    # linear_scaled_min_lr = config.TRAIN.MIN_LR * config.DATA.BATCH_SIZE * dist.get_world_size() / 512.0

    # linear scale the learning rate according to total batch size, may not be optimal
    # linear_scaled_lr = config.TRAIN.BASE_LR * config.DATA.BATCH_SIZE / 512.0
    # linear_scaled_warmup_lr = config.TRAIN.WARMUP_LR * config.DATA.BATCH_SIZE / 512.0
    # linear_scaled_min_lr = config.TRAIN.MIN_LR * config.DATA.BATCH_SIZE / 512.0


    # gradient accumulation also need to scale the learning rate
    if config.TRAIN.ACCUMULATION_STEPS > 1:
        linear_scaled_lr = linear_scaled_lr * config.TRAIN.ACCUMULATION_STEPS
        linear_scaled_warmup_lr = linear_scaled_warmup_lr * config.TRAIN.ACCUMULATION_STEPS
        linear_scaled_min_lr = linear_scaled_min_lr * config.TRAIN.ACCUMULATION_STEPS
    config.defrost()
    config.TRAIN.BASE_LR = linear_scaled_lr
    config.TRAIN.WARMUP_LR = linear_scaled_warmup_lr
    config.TRAIN.MIN_LR = linear_scaled_min_lr
    config.freeze()

    os.makedirs(config.OUTPUT, exist_ok=True)

    #TODO: repair distro training
    logger = create_logger(output_dir=config.OUTPUT, dist_rank=dist.get_rank(), name=f"{config.MODEL.NAME}")
    # logger = create_logger(output_dir=config.OUTPUT, dist_rank=0, name=f"{config.MODEL.NAME}")

    #TODO: repair distro training
    if dist.get_rank() == 0:
        path = os.path.join(config.OUTPUT, "config.json")
        with open(path, "w") as f:
            f.write(config.dump())
        logger.info(f"Full config saved to {path}")
    
    path = os.path.join(config.OUTPUT, "config.json")
    with open(path, "w") as f:
        f.write(config.dump())
    logger.info(f"Full config saved to {path}")

    # print config
    # logger.info(config.dump())
    # logger.info(json.dumps(vars(args)))

    main(config, logger=logger)


if __name__ == '__main__':
    # print(torch.cuda.device_count())
    # print(torch.cuda.current_device())
    # for x in range(torch.cuda.device_count()):
    #     print(torch.cuda.get_device_name(x))
    # torch.cuda.set_device(2)

    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '29501'
    # os.environ['WORLD_SIZE'] = '2'
    # os.environ['LOCAL_RANK'] = '2'
    # os.environ['RANK'] = '0'
    
    # os.environ['CUDA_VISIBLE_DEVICES'] = '5'
    # vis_devs = os.environ['CUDA_VISIBLE_DEVICES'].split(',')
    print(torch.cuda.device_count())
    print([torch.cuda.get_device_properties(x) for x in range(torch.cuda.device_count())])
    
    #vis_devs = [int(x) for x in os.environ['CUDA_VISIBLE_DEVICES'].split(',')]    
    world_size = torch.cuda.device_count()
    #print('CUDA_VISIBLE_DEVICES:', vis_devs)
    #print('world_size:', world_size)

    # torch.multiprocessing.spawn(ddp_main, 
    #                             args=(world_size, [2, 3]),
    #                             nprocs=world_size,
    #                             join=True)

    ddp_main(0, world_size, list(range(world_size)))
