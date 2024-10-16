from models.radar_swin import RadarSwinTransformer
from torch import nn

def build_model(config):
    if config.FUSED_LAYERNORM:
        try:
            import apex as amp
            layernorm = amp.normalization.FusedLayerNorm
            print("Using FusedLayerNorm")
        except:
            layernorm = None
            print("To use FusedLayerNorm, please install apex.")
    else:
        import torch.nn as nn
        layernorm = nn.LayerNorm

    return RadarSwinTransformer(img_size=config.DATA.INPUT_SIZE,
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
                                norm_layer=layernorm,
                                ape=False,
                                patch_norm=False,
                                use_checkpoint=False,
                                fused_window_process=config.FUSED_WINDOW_PROCESS,)