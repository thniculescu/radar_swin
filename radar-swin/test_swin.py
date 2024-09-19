# %%
%load_ext autoreload
%autoreload 2
import argparse
import os
import pprint
import torch

# %% 

from models.radar_swin import RadarSwinTransformer
from data.build import build_loader
from config import get_config
import torch.nn as nn
import torchinfo
import torch.distributed as dist


os.environ["LOCAL_RANK"] = "0"
CONFIG_PATH = "./configs/radarswin/radarswin_tiny.yaml"

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

img_size = config.DATA.INPUT_SIZE
print(img_size)

# dataset_train, dataset_val, data_loader_train, data_loader_val, mixup_fn = build_loader(config)
# print(f"local rank {config.LOCAL_RANK} / global rank {dist.get_rank()} successfully build val dataset")

model = RadarSwinTransformer(img_size=config.DATA.INPUT_SIZE,
                                patch_size=config.MODEL.RADARSWIN.PATCH_SIZE,
                                in_chans=config.MODEL.RADARSWIN.IN_CHANS,
                                num_classes=config.MODEL.NUM_CLASSES,
                                out_reg_heads=config.MODEL.RADARSWIN.OUT_REG_HEADS, #r | center_off | sin_or, cos_or, bb_w, bb_l | vr, vt 
                                embed_dim=config.MODEL.RADARSWIN.EMBED_DIM,
                                depths=config.MODEL.RADARSWIN.DEPTHS,
                                num_heads=config.MODEL.RADARSWIN.NUM_HEADS,
                                window_sizes=config.MODEL.RADARSWIN.WINDOW_SIZES,
                                merge_factors=config.MODEL.RADARSWIN.MERGE_FACTORS,
                                reduction_factors=config.MODEL.RADARSWIN.REDUCTION_FACTORS,
                                mlp_ratio=config.MODEL.RADARSWIN.MLP_RATIO,
                                qkv_bias=config.MODEL.RADARSWIN.QKV_BIAS,
                                qk_scale=config.MODEL.RADARSWIN.QK_SCALE,
                                temporal_attention=config.MODEL.RADARSWIN.TEMPORAL_ATTENTION,
                                drop_rate=config.MODEL.DROP_RATE,
                                attn_drop_rate=0.0,
                                drop_path_rate=config.MODEL.DROP_PATH_RATE,
                                norm_layer=nn.LayerNorm,
                                ape=False,
                                patch_norm=False,
                                use_checkpoint=False,
                                fused_window_process=config.FUSED_WINDOW_PROCESS,)
                    
for key in model.state_dict().keys():
    print(key)

# torchinfo.summary(model, (1, 3, img_size[0], img_size[1]), depth=4, col_names=["input_size", "output_size", "num_params", "mult_adds"])

# import torch.onnx
# torch.onnx.export(model, 
#                 torch.randn(1, 3, img_size[0], img_size[1], requires_grad=False),
#                 f="radarswin_tiny.onnx")

# import onnx
# onnx_program = torch.onnx.export(model, (1, 3, img_size[0], img_size[1]), opset_version=12, verbose=True, export_params=True, do_constant_folding=True, input_names=["input"], output_names=["output"], f="radarswin_tiny.onnx")
# onnx_program.save("radarswin_tiny.onnx")
# %%
