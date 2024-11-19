# --------------------------------------------------------
# Swin Transformer
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu
# --------------------------------------------------------

import os
import time
import torch
import numpy as np
import torch.distributed as dist
# from torchvision import datasets, transforms
# from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
# from timm.data import Mixup
# from timm.data import create_transform
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split


from . import radar_transforms
# from .cached_image_folder import CachedImageFolder
from .samplers import SubsetRandomSampler

# try:
#     from torchvision.transforms import InterpolationMode


#     def _pil_interp(method):
#         if method == 'bicubic':
#             return InterpolationMode.BICUBIC
#         elif method == 'lanczos':
#             return InterpolationMode.LANCZOS
#         elif method == 'hamming':
#             return InterpolationMode.HAMMING
#         else:
#             # default bilinear, do we want to allow nearest?
#             return InterpolationMode.BILINEAR


#     import timm.data.transforms as timm_transforms

#     timm_transforms._pil_interp = _pil_interp
# except:
#     from timm.data.transforms import _pil_interp


def build_loader(config):
    config.defrost()
    dataset_train = build_dataset(is_train=True, config=config)
    config.freeze()
    print(f"local rank {config.LOCAL_RANK} / global rank {dist.get_rank()} successfully build train dataset")
    dataset_val = build_dataset(is_train=False, config=config)
    print(f"local rank {config.LOCAL_RANK} / global rank {dist.get_rank()} successfully build val dataset")

    num_tasks = dist.get_world_size()
    global_rank = dist.get_rank()
    # if config.DATA.ZIP_MODE and config.DATA.CACHE_MODE == 'part':
        # indices = np.arange(dist.get_rank(), len(dataset_train), dist.get_world_size())
        # sampler_train = SubsetRandomSampler(indices)
    # else:

    indices = np.arange(dist.get_rank(), len(dataset_train), dist.get_world_size())
    sampler_train = SubsetRandomSampler(indices)

    # sampler_train = torch.utils.data.DistributedSampler(
    #     dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
    # )

    if config.TEST.SEQUENTIAL:
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)
    else:
        sampler_val = torch.utils.data.distributed.DistributedSampler(
            dataset_val, shuffle=config.TEST.SHUFFLE
        )

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=config.DATA.BATCH_SIZE,
        num_workers=config.DATA.NUM_WORKERS,
        pin_memory=config.DATA.PIN_MEMORY,
        drop_last=True,
    )

    if config.MODEL.TRACKING and config.ALL_SCENE_PARALLEL:
        data_loader_val = torch.utils.data.DataLoader(
            dataset_val, sampler=sampler_val,
            batch_size=170,
            num_workers=config.DATA.NUM_WORKERS,
            pin_memory=config.DATA.PIN_MEMORY,
            drop_last=False,
            shuffle=False,
        )
    else:
        data_loader_val = torch.utils.data.DataLoader(
            dataset_val, sampler=sampler_val,
            batch_size=config.DATA.BATCH_SIZE,
            shuffle=False,
            num_workers=config.DATA.NUM_WORKERS,
            pin_memory=config.DATA.PIN_MEMORY,
            drop_last=False
        )

    # setup mixup / cutmix
    mixup_fn = None
    # mixup_active = config.AUG.MIXUP > 0 or config.AUG.CUTMIX > 0. or config.AUG.CUTMIX_MINMAX is not None
    # if mixup_active:
    #     mixup_fn = Mixup(
    #         mixup_alpha=config.AUG.MIXUP, cutmix_alpha=config.AUG.CUTMIX, cutmix_minmax=config.AUG.CUTMIX_MINMAX,
    #         prob=config.AUG.MIXUP_PROB, switch_prob=config.AUG.MIXUP_SWITCH_PROB, mode=config.AUG.MIXUP_MODE,
    #         label_smoothing=config.MODEL.LABEL_SMOOTHING, num_classes=config.MODEL.NUM_CLASSES)

    return dataset_train, dataset_val, data_loader_train, data_loader_val, mixup_fn


class RadarSwinDataSet(Dataset):
    data = None
    def __init__(self, dataset_root, config, transform=None, target_transform=None, scene_ids=None):
        self.dataset_root = dataset_root
        self.config = config

        if RadarSwinDataSet.data is None:
            RadarSwinDataSet.data = np.load(self.dataset_root, allow_pickle=True).item()

        if scene_ids is None:
            self.scene_ids = list(self.data.keys())
            self.scene_ids.sort()
        else:
            self.scene_ids = [f for f in self.data.keys() 
                                 if f in scene_ids]
            self.scene_ids.sort()

        self.sweeps = [(scene_id, sweep_nr) for scene_id in self.scene_ids for sweep_nr in range(len(RadarSwinDataSet.data[scene_id]['radar']))]


        if (config.MODEL.TRACKING):
            #filter out sweep_nr = 0 for all scenes
            self.sweeps = [(scene_id, sweep_nr) for scene_id, sweep_nr in self.sweeps if sweep_nr != 0]
            #sort self.sweeps by sweep_nr and then scene_id
        if (config.ALL_SCENE_PARALLEL):
            self.sweeps.sort(key=lambda x: (x[1], x[0]))

        # print("sweeps", self.sweeps[:50])

        #shuffle sweeps
        # np.random.shuffle(self.sweeps)

        # print("(scene_id, sweep_id)", self.sweeps)

        self.transform = transform
        self.target_transform = target_transform


    def __getitem__(self, index):
        img = torch.from_numpy(RadarSwinDataSet.data[self.sweeps[index][0]]['radar'][self.sweeps[index][1]])
        anns = torch.from_numpy(RadarSwinDataSet.data[self.sweeps[index][0]]['anns'][self.sweeps[index][1]])
        hmap = torch.from_numpy(RadarSwinDataSet.data[self.sweeps[index][0]]['heatmap'][self.sweeps[index][1]])

        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform((anns, hmap))

        if self.config.MODEL.TRACKING:
            prev_hmap = torch.from_numpy(RadarSwinDataSet.data[self.sweeps[index][0]]['heatmap'][self.sweeps[index][1]-1])
            # stack hmap 6 times:
            prev_hmap = torch.stack([prev_hmap for _ in range(6)], dim=0).to(torch.float16)
            prev_hmap = prev_hmap.unsqueeze(0)
            img = torch.cat((img, prev_hmap), dim=0)


        return img, target, self.sweeps[index]

    def __len__(self):
        return len(self.sweeps)
     

def build_dataset(is_train, config):
    transform, target_transform = build_transform(config)
    if config.DATA.DATASET == 'nuscenes':
        root = os.path.join(config.DATA.DATA_PATH)
        
        # Get all scene IDs

        all_scene_ids = list(np.load(root, allow_pickle=True).item().keys())
        # print("10 all_scene_ids", all_scene_ids[:50])
        # Split scene IDs
        train_scenes, val_scenes = train_test_split(all_scene_ids, test_size=config.DATA.VAL_RATIO, random_state=42)
        
        print(f"NR train:{len(train_scenes)}, NR val:{len(val_scenes)}")
        # print(f"train scenes: {train_scenes}")
        # print(f"val scenes: {val_scenes}")
        
        if is_train:
            dataset = RadarSwinDataSet(root,
                                       config,
                                       transform=transform, 
                                       target_transform=target_transform,
                                       scene_ids=train_scenes,)
        else:
            dataset = RadarSwinDataSet(root, 
                                       config,
                                       transform=transform, 
                                       target_transform=target_transform,
                                       scene_ids=val_scenes,)
            
    else:
        raise NotImplementedError("We only support NuScenes Now.")

    return dataset


def build_transform(config):
    transform = radar_transforms.input_transform(config)
    target_transform = radar_transforms.target_transform(config)
    return transform, target_transform


###########################################################
####################### OLD DATASET #######################
###########################################################


# class RadarSwinDataSetOld(Dataset):
#     def __init__(self, dataset_root, transform=None, target_transform=None, scene_ids=None):
#         self.dataset_root = dataset_root

#         if scene_ids is None:
#             self.sweeps_names = os.listdir(os.path.join(self.dataset_root, "radar"))
#             self.sweeps_names.sort()
#         else:
#             self.sweeps_names = [f for f in os.listdir(os.path.join(self.dataset_root, "radar")) 
#                                  if int(f.split('_')[1]) in scene_ids]
#             self.sweeps_names.sort()

#         self.ann_names = self.sweeps_names
#         self.hmap_names = self.sweeps_names

#         # print("sweeps_names", self.sweeps_names)

#         self.transform = transform
#         self.target_transform = target_transform


#     def __getitem__(self, index):
#         img = torch.from_numpy(np.load(os.path.join(self.dataset_root, "radar", self.sweeps_names[index])))
#         anns = torch.from_numpy(np.load(os.path.join(self.dataset_root, "anns", self.ann_names[index])))
#         hmap = torch.from_numpy(np.load(os.path.join(self.dataset_root, "hmap", self.hmap_names[index])))

#         if self.transform is not None:
#             img = self.transform(img)
#         if self.target_transform is not None:
#             target = self.target_transform((anns, hmap))

#         return img, target

#     def __len__(self):
#         return len(self.sweeps_names)
     

# def build_dataset_old(is_train, config):
#     transform, target_transform = build_transform(config)
#     if config.DATA.DATASET == 'nuscenes':
#         root = os.path.join(config.DATA.DATA_PATH)
        
#         # Get all scene IDs
#         all_sweeps = os.listdir(os.path.join(root, 'radar'))
#         all_scene_ids = sorted(set([int(f.split('_')[1]) for f in all_sweeps]))
        
#         # Split scene IDs
#         train_scenes, val_scenes = train_test_split(all_scene_ids, test_size=config.DATA.VAL_RATIO, random_state=42)
        
#         print(f"NR train:{len(train_scenes)}, NR val:{len(val_scenes)}")
#         # print(f"train scenes: {train_scenes}")
#         # print(f"val scenes: {val_scenes}")
        
#         if is_train:
#             dataset = RadarSwinDataSetOld(root, 
#                                        transform=transform, 
#                                        target_transform=target_transform,
#                                        scene_ids=train_scenes)
#         else:
#             dataset = RadarSwinDataSetOld(root, 
#                                        transform=transform, 
#                                        target_transform=target_transform,
#                                        scene_ids=val_scenes)
#     else:
#         raise NotImplementedError("We only support NuScenes Now.")

#     return dataset


# # def build_transform(config):
# #     transform = radar_transforms.input_transform(config)
# #     target_transform = radar_transforms.target_transform(config)
#         # transform = create_transform(
#         #     input_size=config.DATA.IMG_SIZE,
#         #     is_training=True,
#         #     color_jitter=config.AUG.COLOR_JITTER if config.AUG.COLOR_JITTER > 0 else None,
#         #     auto_augment=config.AUG.AUTO_AUGMENT if config.AUG.AUTO_AUGMENT != 'none' else None,
#         #     re_prob=config.AUG.REPROB,
#         #     re_mode=config.AUG.REMODE,
#         #     re_count=config.AUG.RECOUNT,
#         #     interpolation=config.DATA.INTERPOLATION,
#         # )
#         # if not resize_im:
#         #     # replace RandomResizedCropAndInterpolation with
#         #     # RandomCrop
#         #     transform.transforms[0] = transforms.RandomCrop(config.DATA.IMG_SIZE, padding=4)
#     # return transform, target_transform

#     # t = []
#     # if resize_im:
#     #     if config.TEST.CROP:
#     #         size = int((256 / 224) * config.DATA.IMG_SIZE)
#     #         t.append(
#     #             transforms.Resize(size, interpolation=_pil_interp(config.DATA.INTERPOLATION)),
#     #             # to maintain same ratio w.r.t. 224 images
#     #         )
#     #         t.append(transforms.CenterCrop(config.DATA.IMG_SIZE))
#     #     else:
#     #         t.append(
#     #             transforms.Resize((config.DATA.IMG_SIZE, config.DATA.IMG_SIZE),
#     #                               interpolation=_pil_interp(config.DATA.INTERPOLATION))
#     #         )

#     # t.append(transforms.ToTensor())
#     # t.append(transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD))
#     # return transforms.Compose(t)
