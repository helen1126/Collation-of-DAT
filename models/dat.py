# --------------------------------------------------------
# Swin Transformer
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu
# --------------------------------------------------------
# Vision Transformer with Deformable Attention
# Modified by Zhuofan Xia 
# --------------------------------------------------------

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.models.layers import DropPath, to_2tuple

from .slide import SlideAttention
from .dat_blocks import *
from .nat import NeighborhoodAttention2D
from .qna import FusedKQnA


class LayerScale(nn.Module):

    def __init__(self,
                 dim: int,
                 inplace: bool = False,
                 init_values: float = 1e-5):
        """
        初始化 LayerScale 模块。

        参数:
        dim (int): 输入特征的维度。
        inplace (bool, 可选): 是否进行原地操作。默认为 False。
        init_values (float, 可选): 初始化权重的值。默认为 1e-5。
        """
        super().__init__()
        self.inplace = inplace
        self.weight = nn.Parameter(torch.ones(dim) * init_values)

    def forward(self, x):
        """
        前向传播函数，对输入特征进行缩放。

        参数:
        x (torch.Tensor): 输入的特征图。

        返回:
        torch.Tensor: 缩放后的特征图。
        """
        if self.inplace:
            return x.mul_(self.weight.view(-1, 1, 1))
        else:
            return x * self.weight.view(-1, 1, 1)

class TransformerStage(nn.Module):
    """
    变压器阶段模块，包含多个注意力层和 MLP 层。
    """
    def __init__(self, fmap_size, window_size, ns_per_pt,
                 dim_in, dim_embed, depths, stage_spec, n_groups, 
                 use_pe, sr_ratio,
                 heads, heads_q, stride,
                 offset_range_factor,
                 dwc_pe, no_off, fixed_pe,
                 attn_drop, proj_drop, expansion, drop, drop_path_rate, 
                 use_dwc_mlp, ksize, nat_ksize,
                 k_qna, nq_qna, qna_activation, 
                 layer_scale_value, use_lpu, log_cpb):
        """
        初始化 TransformerStage 模块。

        参数:
        fmap_size (int or tuple): 特征图的大小。
        window_size (int): 窗口大小。
        ns_per_pt (int): 每个点的邻居数量。
        dim_in (int): 输入特征的维度。
        dim_embed (int): 嵌入特征的维度。
        depths (int): 阶段的深度，即层数。
        stage_spec (list): 阶段的配置列表，指定每层的注意力类型。
        n_groups (int): 分组数量。
        use_pe (bool): 是否使用位置编码。
        sr_ratio (int): 下采样率。
        heads (int): 注意力头的数量。
        heads_q (int): 查询注意力头的数量。
        stride (int): 步长。
        offset_range_factor (int): 偏移范围因子。
        dwc_pe (bool): 是否使用深度可分离卷积位置编码。
        no_off (bool): 是否不使用偏移。
        fixed_pe (bool): 是否使用固定位置编码。
        attn_drop (float): 注意力层的丢弃率。
        proj_drop (float): 投影层的丢弃率。
        expansion (int): MLP 层的扩展因子。
        drop (float): 丢弃率。
        drop_path_rate (list): 随机深度丢弃率列表。
        use_dwc_mlp (bool): 是否使用深度可分离卷积 MLP。
        ksize (int): 卷积核大小。
        nat_ksize (int): 邻域注意力的卷积核大小。
        k_qna (int): FusedKQnA 模块的参数。
        nq_qna (int): FusedKQnA 模块的参数。
        qna_activation (str): FusedKQnA 模块的激活函数。
        layer_scale_value (float): 层缩放的值。
        use_lpu (bool): 是否使用局部感知单元。
        log_cpb (bool): 是否使用对数相对位置偏差。
        """
        super().__init__()
        fmap_size = to_2tuple(fmap_size)
        self.depths = depths
        hc = dim_embed // heads
        assert dim_embed == heads * hc
        self.proj = nn.Conv2d(dim_in, dim_embed, 1, 1, 0) if dim_in != dim_embed else nn.Identity()
        self.stage_spec = stage_spec
        self.use_lpu = use_lpu

        self.ln_cnvnxt = nn.ModuleDict(
            {str(d): LayerNormProxy(dim_embed) for d in range(depths) if stage_spec[d] == 'X'}
        )
        self.layer_norms = nn.ModuleList(
            [LayerNormProxy(dim_embed) if stage_spec[d // 2] != 'X' else nn.Identity() for d in range(2 * depths)]
        )

        mlp_fn = TransformerMLPWithConv if use_dwc_mlp else TransformerMLP

        self.mlps = nn.ModuleList(
            [ 
                mlp_fn(dim_embed, expansion, drop) for _ in range(depths)
            ]
        )
        self.attns = nn.ModuleList()
        self.drop_path = nn.ModuleList()
        self.layer_scales = nn.ModuleList(
            [
                LayerScale(dim_embed, init_values=layer_scale_value) if layer_scale_value > 0.0 else nn.Identity() 
                for _ in range(2 * depths)
            ]
        )
        self.local_perception_units = nn.ModuleList(
            [
                nn.Conv2d(dim_embed, dim_embed, kernel_size=3, stride=1, padding=1, groups=dim_embed) if use_lpu else nn.Identity()
                for _ in range(depths)
            ]
        )

        for i in range(depths):
            if stage_spec[i] == 'L':
                self.attns.append(
                    LocalAttention(dim_embed, heads, window_size, attn_drop, proj_drop)
                )
            elif stage_spec[i] == 'D':
                self.attns.append(
                    DAttentionBaseline(fmap_size, fmap_size, heads, 
                    hc, n_groups, attn_drop, proj_drop, 
                    stride, offset_range_factor, use_pe, dwc_pe, 
                    no_off, fixed_pe, ksize, log_cpb)
                )
            elif stage_spec[i] == 'S':
                shift_size = math.ceil(window_size / 2)
                self.attns.append(
                    ShiftWindowAttention(dim_embed, heads, window_size, attn_drop, proj_drop, shift_size, fmap_size)
                )
            elif stage_spec[i] == 'N':
                self.attns.append(
                    NeighborhoodAttention2D(dim_embed, nat_ksize, heads, attn_drop, proj_drop)
                )
            elif stage_spec[i] == 'P':
                self.attns.append(
                    PyramidAttention(dim_embed, heads, attn_drop, proj_drop, sr_ratio)
                )
            elif stage_spec[i] == 'Q':
                self.attns.append(
                    FusedKQnA(nq_qna, dim_embed, heads_q, k_qna, 1, 0, qna_activation)
                )
            elif self.stage_spec[i] == 'X':
                self.attns.append(
                    nn.Conv2d(dim_embed, dim_embed, kernel_size=window_size, padding=window_size // 2, groups=dim_embed)
                )
            elif self.stage_spec[i] == 'E':
                self.attns.append(
                    SlideAttention(dim_embed, heads, 3)
                )
            else:
                raise NotImplementedError(f'Spec: {stage_spec[i]} is not supported.')

            self.drop_path.append(DropPath(drop_path_rate[i]) if drop_path_rate[i] > 0.0 else nn.Identity())

    def forward(self, x):
        """
        前向传播函数，处理输入特征并通过多个注意力层和 MLP 层。

        参数:
        x (torch.Tensor): 输入的特征图。

        返回:
        torch.Tensor: 输出的特征图。
        """
        x = self.proj(x)

        for d in range(self.depths):
            
            if self.use_lpu:
                x0 = x
                x = self.local_perception_units[d](x.contiguous())
                x = x + x0

            if self.stage_spec[d] == 'X':
                x0 = x
                x = self.attns[d](x)
                x = self.mlps[d](self.ln_cnvnxt[str(d)](x))
                x = self.drop_path[d](x) + x0
            else:
                x0 = x
                x, pos, ref = self.attns[d](self.layer_norms[2 * d](x))
                x = self.layer_scales[2 * d](x)
                x = self.drop_path[d](x) + x0
                x0 = x
                x = self.mlps[d](self.layer_norms[2 * d + 1](x))
                x = self.layer_scales[2 * d + 1](x)
                x = self.drop_path[d](x) + x0

        return x


class DAT(nn.Module):
    """
    具有可变形注意力的视觉变压器模型。
    """
    def __init__(self, img_size=224, patch_size=4, num_classes=1000, expansion=4,
                 dim_stem=96, dims=[96, 192, 384, 768], depths=[2, 2, 6, 2], 
                 heads=[3, 6, 12, 24], heads_q=[6, 12, 24, 48],
                 window_sizes=[7, 7, 7, 7],
                 drop_rate=0.0, attn_drop_rate=0.0, drop_path_rate=0.0, 
                 strides=[-1,-1,-1,-1],
                 offset_range_factor=[1, 2, 3, 4],
                 stage_spec=[['L', 'D'], ['L', 'D'], ['L', 'D', 'L', 'D', 'L', 'D'], ['L', 'D']], 
                 groups=[-1, -1, 3, 6],
                 use_pes=[False, False, False, False], 
                 dwc_pes=[False, False, False, False],
                 sr_ratios=[8, 4, 2, 1], 
                 lower_lr_kvs={},
                 fixed_pes=[False, False, False, False],
                 no_offs=[False, False, False, False],
                 ns_per_pts=[4, 4, 4, 4],
                 use_dwc_mlps=[False, False, False, False],
                 use_conv_patches=False,
                 ksizes=[9, 7, 5, 3],
                 ksize_qnas=[3, 3, 3, 3],
                 nqs=[2, 2, 2, 2],
                 qna_activation='exp',
                 nat_ksizes=[3,3,3,3],
                 layer_scale_values=[-1,-1,-1,-1],
                 use_lpus=[False, False, False, False],
                 log_cpb=[False, False, False, False],
                 **kwargs):
        """
        初始化 DAT 模型。

        参数:
        img_size (int, 可选): 输入图像的大小。默认为 224。
        patch_size (int, 可选): 图像块的大小。默认为 4。
        num_classes (int, 可选): 分类的类别数量。默认为 1000。
        expansion (int, 可选): MLP 层的扩展因子。默认为 4。
        dim_stem (int, 可选): 茎层的维度。默认为 96。
        dims (list, 可选): 各阶段的维度列表。默认为 [96, 192, 384, 768]。
        depths (list, 可选): 各阶段的深度列表。默认为 [2, 2, 6, 2]。
        heads (list, 可选): 各阶段的注意力头数量列表。默认为 [3, 6, 12, 24]。
        heads_q (list, 可选): 各阶段的查询注意力头数量列表。默认为 [6, 12, 24, 48]。
        window_sizes (list, 可选): 各阶段的窗口大小列表。默认为 [7, 7, 7, 7]。
        drop_rate (float, 可选): 丢弃率。默认为 0.0。
        attn_drop_rate (float, 可选): 注意力层的丢弃率。默认为 0.0。
        drop_path_rate (float, 可选): 随机深度丢弃率。默认为 0.0。
        strides (list, 可选): 各阶段的步长列表。默认为 [-1,-1,-1,-1]。
        offset_range_factor (list, 可选): 各阶段的偏移范围因子列表。默认为 [1, 2, 3, 4]。
        stage_spec (list, 可选): 各阶段的配置列表。默认为 [['L', 'D'], ['L', 'D'], ['L', 'D', 'L', 'D', 'L', 'D'], ['L', 'D']]。
        groups (list, 可选): 各阶段的分组数量列表。默认为 [-1, -1, 3, 6]。
        use_pes (list, 可选): 各阶段是否使用位置编码的列表。默认为 [False, False, False, False]。
        dwc_pes (list, 可选): 各阶段是否使用深度可分离卷积位置编码的列表。默认为 [False, False, False, False]。
        sr_ratios (list, 可选): 各阶段的下采样率列表。默认为 [8, 4, 2, 1]。
        lower_lr_kvs (dict, 可选): 低学习率的键值对。默认为 {}。
        fixed_pes (list, 可选): 各阶段是否使用固定位置编码的列表。默认为 [False, False, False, False]。
        no_offs (list, 可选): 各阶段是否不使用偏移的列表。默认为 [False, False, False, False]。
        ns_per_pts (list, 可选): 各阶段每个点的邻居数量列表。默认为 [4, 4, 4, 4]。
        use_dwc_mlps (list, 可选): 各阶段是否使用深度可分离卷积 MLP 的列表。默认为 [False, False, False, False]。
        use_conv_patches (bool, 可选): 是否使用卷积图像块。默认为 False。
        ksizes (list, 可选): 各阶段的卷积核大小列表。默认为 [9, 7, 5, 3]。
        ksize_qnas (list, 可选): 各阶段 FusedKQnA 模块的卷积核大小列表。默认为 [3, 3, 3, 3]。
        nqs (list, 可选): 各阶段 FusedKQnA 模块的参数列表。默认为 [2, 2, 2, 2]。
        qna_activation (str, 可选): FusedKQnA 模块的激活函数。默认为 'exp'。
        nat_ksizes (list, 可选): 各阶段邻域注意力的卷积核大小列表。默认为 [3,3,3,3]。
        layer_scale_values (list, 可选): 各阶段的层缩放值列表。默认为 [-1,-1,-1,-1]。
        use_lpus (list, 可选): 各阶段是否使用局部感知单元的列表。默认为 [False, False, False, False]。
        log_cpb (list, 可选): 各阶段是否使用对数相对位置偏差的列表。默认为 [False, False, False, False]。
        """
        super().__init__()

        self.patch_proj = nn.Sequential(
            nn.Conv2d(3, dim_stem // 2, 3, patch_size // 2, 1),
            LayerNormProxy(dim_stem // 2),
            nn.GELU(),
            nn.Conv2d(dim_stem // 2, dim_stem, 3, patch_size // 2, 1),
            LayerNormProxy(dim_stem)
        ) if use_conv_patches else nn.Sequential(
            nn.Conv2d(3, dim_stem, patch_size, patch_size, 0),
            LayerNormProxy(dim_stem)
        )

        img_size = img_size // patch_size
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        self.stages = nn.ModuleList()
        for i in range(4):
            dim1 = dim_stem if i == 0 else dims[i - 1] * 2
            dim2 = dims[i]
            self.stages.append(
                TransformerStage(
                    img_size, window_sizes[i], ns_per_pts[i],
                    dim1, dim2, depths[i],
                    stage_spec[i], groups[i], use_pes[i],
                    sr_ratios[i], heads[i], heads_q[i], strides[i],
                    offset_range_factor[i],
                    dwc_pes[i], no_offs[i], fixed_pes[i],
                    attn_drop_rate, drop_rate, expansion, drop_rate,
                    dpr[sum(depths[:i]):sum(depths[:i + 1])], use_dwc_mlps[i],
                    ksizes[i], nat_ksizes[i], ksize_qnas[i], nqs[i],qna_activation,
                    layer_scale_values[i], use_lpus[i], log_cpb[i]
                )
            )
            img_size = img_size // 2

        self.down_projs = nn.ModuleList()
        for i in range(3):
            self.down_projs.append(
                nn.Sequential(
                    nn.Conv2d(dims[i], dims[i + 1], 3, 2, 1, bias=False),
                    LayerNormProxy(dims[i + 1])
                ) if use_conv_patches else nn.Sequential(
                    nn.Conv2d(dims[i], dims[i + 1], 2, 2, 0, bias=False),
                    LayerNormProxy(dims[i + 1])
                )
            )

        self.cls_norm = LayerNormProxy(dims[-1]) 
        self.cls_head = nn.Linear(dims[-1], num_classes)

        self.lower_lr_kvs = lower_lr_kvs

        self.reset_parameters()

    def reset_parameters(self):
        """
        重置模型的参数。

        该函数遍历模型的所有参数，对于类型为 nn.Linear 或 nn.Conv2d 的参数，使用 Kaiming 正态分布初始化权重，
        并将偏置初始化为零。
        """
        for m in self.parameters():
            if isinstance(m, (nn.Linear, nn.Conv2d)):
                nn.init.kaiming_normal_(m.weight)
                nn.init.zeros_(m.bias)

    @torch.no_grad()
    def load_pretrained(self, state_dict, lookup_22k):
        """
        加载预训练模型的权重。

        该函数将预训练模型的权重加载到当前模型中，处理了形状不匹配的情况，如相对位置索引、q_grid、reference 等，
        对于相对位置偏差表和 rpe_table 进行双三次插值，对于分类头根据 lookup_22k 进行处理。

        参数:
        state_dict (dict): 预训练模型的状态字典。
        lookup_22k (list or tensor): 用于处理分类头的查找表。

        返回:
        torch.nn.modules.module._IncompatibleKeys: 加载状态字典时的不兼容键信息。
        """
        new_state_dict = {}
        for state_key, state_value in state_dict.items():
            keys = state_key.split('.')
            m = self
            for key in keys:
                if key.isdigit():
                    m = m[int(key)]
                else:
                    m = getattr(m, key)
            if m.shape == state_value.shape:
                new_state_dict[state_key] = state_value
            else:
                # Ignore different shapes
                if 'relative_position_index' in keys:
                    new_state_dict[state_key] = m.data
                if 'q_grid' in keys:
                    new_state_dict[state_key] = m.data
                if 'reference' in keys:
                    new_state_dict[state_key] = m.data
                # Bicubic Interpolation
                if 'relative_position_bias_table' in keys:
                    n, c = state_value.size()
                    l_side = int(math.sqrt(n))
                    assert n == l_side ** 2
                    L = int(math.sqrt(m.shape[0]))
                    pre_interp = state_value.reshape(1, l_side, l_side, c).permute(0, 3, 1, 2)
                    post_interp = F.interpolate(pre_interp, (L, L), mode='bicubic')
                    new_state_dict[state_key] = post_interp.reshape(c, L ** 2).permute(1, 0)
                if 'rpe_table' in keys:
                    c, h, w = state_value.size()
                    C, H, W = m.data.size()
                    pre_interp = state_value.unsqueeze(0)
                    post_interp = F.interpolate(pre_interp, (H, W), mode='bicubic')
                    new_state_dict[state_key] = post_interp.squeeze(0)
                if 'cls_head' in keys:
                    new_state_dict[state_key] = state_value[lookup_22k]

        msg = self.load_state_dict(new_state_dict, strict=False)
        return msg

    @torch.jit.ignore
    def no_weight_decay(self):
        """
        返回不需要进行权重衰减的参数名称集合。

        返回:
        set: 不需要进行权重衰减的参数名称集合，当前仅包含 'absolute_pos_embed'。
        """
        return {'absolute_pos_embed'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        """
        返回不需要进行权重衰减的参数关键字集合。

        返回:
        set: 不需要进行权重衰减的参数关键字集合，当前包含 'relative_position_bias_table' 和 'rpe_table'。
        """
        return {'relative_position_bias_table', 'rpe_table'}

    def forward(self, x):
        """
        前向传播函数。

        该函数定义了输入数据在模型中的前向传播过程，包括 patch 投影、多个 Transformer 阶段的处理、下采样、归一化、
        平均池化和分类头处理。

        参数:
        x (torch.Tensor): 输入的图像数据，形状为 (batch_size, channels, height, width)。

        返回:
        tuple: 包含三个元素，第一个元素是模型的输出结果，后两个元素当前为 None。
        """
        x = self.patch_proj(x)
        for i in range(4):
            x = self.stages[i](x)
            if i < 3:
                x = self.down_projs[i](x)
        x = self.cls_norm(x)
        x = F.adaptive_avg_pool2d(x, 1)
        x = torch.flatten(x, 1)
        x = self.cls_head(x)
        return x, None, None
