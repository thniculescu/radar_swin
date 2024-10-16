# --------------------------------------------------------
# Swin Transformer
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu
# --------------------------------------------------------

import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
from timm.models.layers import DropPath, to_2tuple, trunc_normal_

# try:
#     import os, sys

#     kernel_path = os.path.abspath(os.path.join('..'))
#     sys.path.append(kernel_path)
#     from kernels.window_process.window_process import WindowProcess, WindowProcessReverse

# except:
#     WindowProcess = None
#     WindowProcessReverse = None
#     print("[Warning] Fused window process have not been installed. Please refer to get_started.md for installation.")


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size (tuple(int)): window size

    Returns:
        windows: (num_windows*B, window_size[0], window_size[1], C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size[0], window_size[0], W // window_size[1], window_size[1], C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size[0], window_size[1], C)
    return windows


def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: (num_windows*B, window_size[0], window_size[1], C)
        window_size (tuple(int)): window size
        H (int): Height of image
        W (int): Width of image

    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size[0] / window_size[1]))
    x = windows.view(B, H // window_size[0], W // window_size[1], window_size[0], window_size[1], -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class RadarWindowAttention(nn.Module):
    r""" Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.

    Args:
        dim (int): Number of input channels.
        window_size (tuple(int)): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):

        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing='ij'))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4) #!! 3, B_, nH, N, C//nH
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        attn = attn + relative_position_bias.unsqueeze(0) #!! B_, nH, N, N

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def extra_repr(self) -> str:
        return f'dim={self.dim}, window_size={self.window_size}, num_heads={self.num_heads}'

    def flops(self, N):
        # calculate flops for 1 window with token length of N
        flops = 0
        # qkv = self.qkv(x)
        flops += N * self.dim * 3 * self.dim
        # attn = (q @ k.transpose(-2, -1))
        flops += self.num_heads * N * (self.dim // self.num_heads) * N
        #  x = (attn @ v)
        flops += self.num_heads * N * N * (self.dim // self.num_heads)
        # x = self.proj(x)
        flops += N * self.dim * self.dim
        return flops


class RadarSwinTransformerBlock(nn.Module):
    r""" Swin Transformer Block.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple(int)): Input resolution. (Nsweeps*Nbins)
        num_heads (int): Number of attention heads.
        window_size (tuple(int)): Window size (Wh, Ww)
        shift_size (tuple(int)): Shift size for SW-MSA. (Sh, Sw). Default: (0, 0)
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
        fused_window_process (bool, optional): If True, use one kernel to fused window shift & window partition for acceleration, similar for the reversed part. Default: False
    """

    def __init__(self, dim, input_resolution, num_heads, window_size=(2, 2), shift_size=(0, 0),
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                 fused_window_process=False, temporal_attention=False):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        self.temporal_attention = temporal_attention

        # if min(self.input_resolution) <= self.window_size:
        #     # if window size is larger than input resolution, we don't partition windows
        #     self.shift_size = 0
        #     self.window_size = min(self.input_resolution)
        assert 0 <= self.shift_size[0] < self.window_size[0], "shift_size must in 0-window_size"
        assert 0 <= self.shift_size[1] < self.window_size[1], "shift_size must in 0-window_size"

        # print(self.extra_repr())

        self.norm1 = norm_layer(dim)
        self.attn = RadarWindowAttention(
            dim, window_size=self.window_size, num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        H, W = self.input_resolution
        Wh, Ww = self.window_size

        if self.shift_size[0] > 0 or self.shift_size[1] > 0:
            # calculate attention mask for SW-MSA
            img_mask = torch.zeros((1, H, W, 1))  # 1 H W 1
            h_slices = (slice(0, -self.window_size[0]),
                        slice(-self.window_size[0], -self.shift_size[0]),
                        slice(-self.shift_size[0], None))
            w_slices = (slice(0, self.input_resolution[1]),)
                        # slice(-self.window_size, -self.shift_size),
                        # slice(-self.shift_size, None))
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1

            mask_windows = window_partition(img_mask, self.window_size)  # nW, Wh, Ww, 1
            mask_windows = mask_windows.view(-1, self.window_size[0] * self.window_size[1])
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2) # nW, Wh*Ww, Wh*Ww
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        else:
            attn_mask = None

        if self.temporal_attention:
            temp_attn_mask = torch.triu(torch.ones((Wh, Wh))).reshape(Wh, Wh, 1, 1)
            temp_attn_mask = temp_attn_mask.expand(-1, -1, Ww, Ww).permute(0, 2, 1, 3).reshape(Wh * Ww, Wh * Ww) # Wh*Ww, Wh*Ww
            temp_attn_mask = temp_attn_mask.masked_fill(temp_attn_mask == 0, float(-100.0)).masked_fill(temp_attn_mask != 0, float(0.0)) # Wh*Ww, Wh*Ww
            if attn_mask is not None:
                attn_mask = attn_mask + temp_attn_mask # nW, Wh*Ww, Wh*Ww
                attn_mask = torch.masked_fill(attn_mask, attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
            else:
                attn_mask = temp_attn_mask.unsqueeze(0).expand(((H // Wh) * (W // Ww)), -1, -1)

        self.register_buffer("attn_mask", attn_mask)

        self.fused_window_process = fused_window_process

    def forward(self, x):
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)

        # cyclic shift
        if self.shift_size[0] > 0 or self.shift_size[1] > 0:
            # if not self.fused_window_process:
            shifted_x = torch.roll(x, shifts=(-self.shift_size[0], -self.shift_size[1]), dims=(1, 2))
            # partition windows
            x_windows = window_partition(shifted_x, self.window_size)  # nW*B, Wh, Ww, C
            # else: TODO: CHECK THIS
            #     x_windows = WindowProcess.apply(x, B, H, W, C, -self.shift_size, self.window_size)
        else:
            shifted_x = x
            # partition windows
            x_windows = window_partition(shifted_x, self.window_size)  # nW*B, Wh, Ww, C

        x_windows = x_windows.view(-1, self.window_size[0] * self.window_size[1], C)  # nW*B, Wh*Ww, C

        # W-MSA/SW-MSA
        attn_windows = self.attn(x_windows, mask=self.attn_mask)  # nW*B, Wh*Ww, C

        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size[0], self.window_size[1], C)

        # reverse cyclic shift
        if self.shift_size[0] > 0 or self.shift_size[1] > 0:
            # if not self.fused_window_process:
            shifted_x = window_reverse(attn_windows, self.window_size, H, W)  # B H' W' C
            x = torch.roll(shifted_x, shifts=(self.shift_size[0], self.shift_size[1]), dims=(1, 2))
            # else: TODO: CHECK THIS
            #     x = WindowProcessReverse.apply(attn_windows, B, H, W, C, self.shift_size, self.window_size)
        else:
            shifted_x = window_reverse(attn_windows, self.window_size, H, W)  # B H' W' C
            x = shifted_x
        x = x.view(B, H * W, C)
        x = shortcut + self.drop_path(x)

        # FFN
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, num_heads={self.num_heads}, " \
               f"window_size={self.window_size}, shift_size={self.shift_size}, mlp_ratio={self.mlp_ratio}"

    def flops(self):
        flops = 0
        H, W = self.input_resolution
        # norm1
        flops += self.dim * H * W
        # W-MSA/SW-MSA
        nW = H * W / self.window_size[0] / self.window_size[1]
        flops += nW * self.attn.flops(self.window_size[0] * self.window_size[1])
        # mlp
        flops += 2 * H * W * self.dim * self.dim * self.mlp_ratio
        # norm2
        flops += self.dim * H * W
        return flops


class PatchMerging(nn.Module):
    r""" Patch Merging Layer.

    Args:
        input_resolution (tuple(int)): Resolution of input feature.
        patch_merge_factor (tuple(int)): Resolution of each patch.
        dim (int): Number of input channels.
        reduction_factor (int): Channel reduction factor for patch merging.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, input_resolution, patch_merge_factor, dim, reduction_factor, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.patch_merge_factor = patch_merge_factor
        self.dim = dim
        self.reduction_factor = reduction_factor

        self.patch_size = int(patch_merge_factor[0] * patch_merge_factor[1])
        self.reduction = nn.Linear(int(self.patch_size * dim), int(self.patch_size * dim // reduction_factor), bias=False)
        self.norm = norm_layer(self.patch_size * dim)

    def forward(self, x):
        """
        x: B, H*W, C
        """
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"
        assert H % self.patch_merge_factor[0] == 0 and W % self.patch_merge_factor[1] == 0, f"x size ({H}*{W}) are not even."

        x = x.view(B, H, W, C)
        x = x.view(B, H // self.patch_merge_factor[0], self.patch_merge_factor[0], W // self.patch_merge_factor[1], self.patch_merge_factor[1], C)
        # x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H // self.patch_merge_factor[0], W // self.patch_merge_factor[1], -1) #TODO: CHECK IS OK
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H * W // self.patch_merge_factor[0] // self.patch_merge_factor[1], -1)

        #old patch merge
        # x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
        # x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
        # x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
        # x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C
        # x = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C
        # x = x.view(B, -1, 4 * C)  # B H/2*W/2 4*C

        x = self.norm(x)
        x = self.reduction(x)

        return x

    def extra_repr(self) -> str:
        return f"input_resolution={self.input_resolution}, patch_merge_factor={self.patch_merge_factor}, dim={self.dim}, reduction_factor={self.reduction_factor}"

    def flops(self):
        H, W = self.input_resolution
        Hf, Wf = self.patch_merge_factor
        flops = H * W * self.dim
        flops += (H // Hf) * (W // Wf) * self.patch_size * self.dim * (self.patch_size / self.reduction_factor) * self.dim
        return flops


class BasicLayer(nn.Module):
    """ A basic Swin Transformer layer for one stage.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple(int)): Input resolution.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        window_size (tuple(int)): Local window size.
        vert_win_shift: If True, use vertical window shift. Default: False
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        temporal_attention (bool, optional): If True, add temporal attention. Default: False
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
        fused_window_process (bool, optional): If True, use one kernel to fused window shift & window partition for acceleration, similar for the reversed part. Default: False
    """

    def __init__(self, dim, input_resolution, depth, num_heads, window_size, vert_win_shift=False,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, temporal_attention=False, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, downsample=None, use_checkpoint=False,
                 fused_window_process=False):

        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        assert depth % 2 == 0, 'BasicLayer should only be used with depth that can be divided by 2 for W-MSA, SW-MSA'
        self.use_checkpoint = use_checkpoint
        self.vert_win_shift = vert_win_shift
        self.temporal_attention = temporal_attention

        # build blocks
        self.blocks = nn.ModuleList([
            RadarSwinTransformerBlock(dim=dim, input_resolution=input_resolution,
                                 num_heads=num_heads, window_size=window_size,
                                 shift_size=(0, 0) if (i % 2 == 0) else ((1 if vert_win_shift else 0) * window_size[0] // 2, window_size[1] // 2),
                                 mlp_ratio=mlp_ratio,
                                 qkv_bias=qkv_bias, qk_scale=qk_scale,
                                 temporal_attention=self.temporal_attention,
                                 drop=drop, attn_drop=attn_drop,
                                 drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                                 norm_layer=norm_layer,
                                 fused_window_process=fused_window_process,)
            for i in range(depth)])

        # patch merging layer
        if downsample is not None:
            # self.downsample = downsample(input_resolution, dim=dim, norm_layer=norm_layer)
            self.downsample = downsample
        else:
            self.downsample = None

    def forward(self, x):
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x)
        if self.downsample is not None:
            x = self.downsample(x)
        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, depth={self.depth}"

    def flops(self):
        flops = 0
        for blk in self.blocks:
            flops += blk.flops()
        if self.downsample is not None:
            flops += self.downsample.flops()
        return flops


class PatchEmbed(nn.Module):
    r""" Image to Patch Embedding

    Args:
        img_size (tuple(int)): Image size.  Default: (6, 360).
        patch_size (tuple(int)): Patch token size. Default: (1, 1).
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 72.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self, img_size=(6, 360), patch_size=(1, 1), in_chans=3, embed_dim=72, norm_layer=None):
        super().__init__()
        # img_size = to_2tuple(img_size)
        # patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        B, C, H, W = x.shape # TODO shape input dataa
        # FIXME look at relaxing size constraints
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x).flatten(2).transpose(1, 2)  # B Ph*Pw C
        if self.norm is not None:
            x = self.norm(x)
        return x

    def flops(self):
        Ho, Wo = self.patches_resolution
        flops = Ho * Wo * self.embed_dim * self.in_chans * (self.patch_size[0] * self.patch_size[1])
        if self.norm is not None:
            flops += Ho * Wo * self.embed_dim
        return flops


class RadarSwinTransformer(nn.Module):
    r""" Swin Transformer
        A PyTorch impl of : `Swin Transformer: Hierarchical Vision Transformer using Shifted Windows`  -
          https://arxiv.org/pdf/2103.14030

    Args:
        img_size (tuple(int)): Input image size. Default (6, 360)
        patch_size (tuple(int)): Patch size. Default: (1, 1)
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1
        out_reg_heads (tuple(int)): Number of regression heads. Default: [1, 1, 4, 2]
        embed_dim (int): Patch embedding dimension. Default: 6
        depths (tuple(int)): Depth of each Swin Transformer layer.
        num_heads (tuple(int)): Number of attention heads in different layers.
        window_sizes (tuple(tuple(int))): Window size in different layers.
        merge_factors (tuple(tuple(int) | None)): Patch merging factor in different layers.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4
        qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float): Override default qk scale of head_dim ** -0.5 if set. Default: None
        temporal_attention (bool): If True, add temporal attention. Default: True
        drop_rate (float): Dropout rate. Default: 0
        attn_drop_rate (float): Attention dropout rate. Default: 0
        drop_path_rate (float): Stochastic depth rate. Default: 0.1
        norm_layer (nn.Module): Normalization layer. Default: nn.LayerNorm.
        ape (bool): If True, add absolute position embedding to the patch embedding. Default: False
        # patch_norm (bool): If True, add normalization after patch embedding. Default: True
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False
        fused_window_process (bool, optional): If True, use one kernel to fused window shift & window partition for acceleration, similar for the reversed part. Default: False
    """

    def __init__(self, img_size=(6, 360), patch_size=(1, 1), in_chans=3, num_classes=1, out_reg_heads=[1, 1, 4, 2],
                 embed_dim=72, depths=[2, 2, 6, 2], num_heads=[3, 6, 6, 12],
                 window_sizes=[(6, 6), (6, 6), (3, 3), (1, 2)], 
                 merge_factors=[(1, 2), (2, 2), (3, 2), (1, 1)], reduction_factors=[1, 2, 2, 1],
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, temporal_attention=True,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm, ape=False, patch_norm=True,
                 use_checkpoint=False, fused_window_process=False, **kwargs):
        super().__init__()

        self.num_classes = num_classes
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.ape = ape
        self.patch_norm = patch_norm

        self.window_sizes = window_sizes
        self.merge_factors = merge_factors
        self.reduction_factors = reduction_factors
        self.out_reg_heads = out_reg_heads

        # print(self.num_layers)
        # print(num_heads)
        # print(window_sizes)
        # print(merge_factors)
        # print(reduction_factors)
        # print(num_classes)

        assert len(num_heads) == self.num_layers, "number of num heads mismatch"
        assert len(window_sizes) == self.num_layers, "number of window size mismatch"
        assert len(merge_factors) == self.num_layers, "number of merge factors mismatch"
        assert len(reduction_factors) == self.num_layers, "number of reduction factors mismatch"
        assert merge_factors[-1] == [1, 1], "last merge factors should be (1, 1)"
        assert reduction_factors[-1] == 1, "last reduction factors should be 1"
        assert len(out_reg_heads) > 0, "at least one output"
        assert num_classes == 1, "num_classes != 1 not implemented"

        self.num_features = [embed_dim]
        for d in range(self.num_layers):
            assert (self.num_features[-1] * merge_factors[d][0] * merge_factors[d][1]) % reduction_factors[d] == 0,\
                f"dim {self.num_features} should be divided by reduction_factors {reduction_factors[d]}"
            self.num_features.append((self.num_features[-1] * merge_factors[d][0] * merge_factors[d][1]) // reduction_factors[d])

        self.mlp_ratio = mlp_ratio

        self.input_resolutions = [img_size]
        for d in range(self.num_layers):
            assert self.input_resolutions[-1][0] % merge_factors[d][0] == 0 and self.input_resolutions[-1][1] % merge_factors[d][1] == 0, \
                f"input resolution {self.input_resolutions[-1]} should be divided by merge factors {merge_factors[d]}"
            self.input_resolutions.append((self.input_resolutions[-1][0] // merge_factors[d][0],
                                           self.input_resolutions[-1][1] // merge_factors[d][1]))

        #check patch size
        if patch_size == (0, 0):
            self.patch_embed = nn.Identity()
        else:
            # split image into non-overlapping patches
            self.patch_embed = PatchEmbed(
                img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim,
                norm_layer=norm_layer if self.patch_norm else None)
            num_patches = self.patch_embed.num_patches
            patches_resolution = self.patch_embed.patches_resolution
            self.patches_resolution = patches_resolution

        # absolute position embedding
        if self.ape:
            self.absolute_pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
            trunc_normal_(self.absolute_pos_embed, std=.02)

        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        # build layers
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = BasicLayer(dim=int(self.num_features[i_layer]),
                               input_resolution=self.input_resolutions[i_layer],
                               depth=depths[i_layer],
                               num_heads=num_heads[i_layer],
                               window_size=self.window_sizes[i_layer],
                               vert_win_shift=False,
                               mlp_ratio=self.mlp_ratio,
                               qkv_bias=qkv_bias, qk_scale=qk_scale, temporal_attention=temporal_attention,
                               drop=drop_rate, attn_drop=attn_drop_rate,
                               drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                               norm_layer=norm_layer,
                               downsample=PatchMerging(self.input_resolutions[i_layer],
                                                        self.merge_factors[i_layer],
                                                        self.num_features[i_layer],
                                                        self.reduction_factors[i_layer],
                                                        norm_layer=norm_layer) if (i_layer < self.num_layers - 1) else None,
                               use_checkpoint=use_checkpoint,
                               fused_window_process=fused_window_process)
            self.layers.append(layer)

        self.norm = norm_layer(self.num_features[-1])
        # self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.reg_outputs = nn.ModuleList()
        #special case for the first regression heads: heatmap, range
        self.reg_outputs.append( #HMAP
            nn.Sequential(
                # nn.Linear(self.num_features[-1], int(mlp_ratio * self.num_features[-1])),
                # nn.GELU(),
                # nn.Dropout(drop_rate),
                # nn.Linear(int(mlp_ratio * self.num_features[-1]), out_reg_heads[0]),
                # nn.Sigmoid()
                nn.Linear(self.num_features[-1], out_reg_heads[0]),
                nn.Sigmoid()
            ))
        self.reg_outputs.append( #RANGE
            nn.Sequential(
                # nn.Linear(self.num_features[-1], int(mlp_ratio * self.num_features[-1])),
                # nn.GELU(),
                # nn.Dropout(drop_rate),
                # nn.Linear(int(mlp_ratio * self.num_features[-1]), out_reg_heads[1]),
                # nn.ReLU()
                nn.Linear(self.num_features[-1], out_reg_heads[1]),
                nn.ReLU()
            ))
        self.reg_outputs.append( #sin cos - relative to angle
            nn.Sequential(
                nn.Linear(self.num_features[-1], out_reg_heads[2]),
                nn.Tanh()
            ))
        
        self.reg_outputs.append( #w l - relative to range
            nn.Sequential(
                nn.Linear(self.num_features[-1], out_reg_heads[3]),
                nn.ReLU()
            ))
        
        self.reg_outputs.append( #Vr Vt - relative to range
            nn.Sequential(
                nn.Linear(self.num_features[-1], out_reg_heads[3]),
                nn.Tanh()
            ))
        


        # for i in range(len(out_reg_heads)):
        #     self.reg_outputs.append(nn.Linear(self.num_features[-1], out_reg_heads[i]))

        # self.head = nn.Linear(self.num_features[-1], num_classes) if num_classes > 0 else nn.Identity()

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'absolute_pos_embed'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table'}

    def forward_features(self, x):
        if self.ape:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)

        for layer in self.layers:
            x = layer(x)

        return x
    
    def forward_reg_heads(self, x):
        #create a tensor from applying the different regression heads, so that torchsummary can see it
        x = torch.cat([reg_head(x) for reg_head in self.reg_outputs], dim=1)
        return x

    def forward(self, x):
        x = self.patch_embed(x)
        x = self.forward_features(x)

        x = self.norm(x)  # B L C
        # x = self.avgpool(x.transpose(1, 2))  # B C 1
        # x = torch.flatten(x, 1)

        B, L, C = x.shape
        x = x.view(B * L, self.num_features[-1])
        x = self.forward_reg_heads(x)
        x = x.view(B, L, -1)

        return x

    def flops(self):
        flops = 0
        flops += self.patch_embed.flops() #PatchEmbed
        for i, layer in enumerate(self.layers): #RadarSwinTransformerBlocks
            flops += layer.flops()
        
        # LayerNorm
        flops += self.num_features[-1] * self.input_resolutions[-1][0] * self.input_resolutions[-1][1] * self.num_classes
        # Linear RegHeads
        flops += self.num_features[-1] * self.input_resolutions[-1][0] * self.input_resolutions[-1][1] * self.num_classes * torch.sum(torch.tensor(self.out_reg_heads))
        return flops
