# --------------------------------------------------------
# Slide-Transformer: Hierarchical Vision Transformer with Local Self-Attention
# Originally written by Xuran Pan
# --------------------------------------------------------
# Vision Transformer with Deformable Attention
# Modified by Zhuofan Xia 
# --------------------------------------------------------

import torch
import torch.nn as nn
import einops
from timm.models.layers import trunc_normal_


class SlideAttention(nn.Module):

    def __init__(
        self, dim, num_heads, ka, qkv_bias=True, qk_scale=None, attn_drop=0.,
        proj_drop=0.,dim_reduction=4, rpb=True, padding_mode='zeros',
        share_dwc_kernel=True, share_qkv=False):
        """
        初始化 SlideAttention 模块。

        参数:
        dim (int): 输入特征的维度。
        num_heads (int): 注意力头的数量。
        ka (int): 卷积核的大小。
        qkv_bias (bool, 可选): 是否在 QKV 映射中使用偏置，默认为 True。
        qk_scale (float, 可选): 查询键缩放因子，默认为 None，此时使用头维度的平方根的倒数。
        attn_drop (float, 可选): 注意力分数的丢弃率，默认为 0。
        proj_drop (float, 可选): 投影层的丢弃率，默认为 0。
        dim_reduction (int, 可选): 维度缩减因子，默认为 4。
        rpb (bool, 可选): 是否使用相对位置偏差，默认为 True。
        padding_mode (str, 可选): 卷积填充模式，默认为 'zeros'。
        share_dwc_kernel (bool, 可选): 是否共享深度可分离卷积核，默认为 True。
        share_qkv (bool, 可选): 是否共享 QKV 映射，默认为 False。
        """
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        if share_qkv:
            self.qkv_scale = 1
        else:
            self.qkv_scale = 3
        self.rpb = rpb
        self.share_dwc_kernel = share_dwc_kernel
        self.padding_mode = padding_mode
        self.share_qkv = share_qkv
        self.ka = ka
        self.dim_reduction = dim_reduction
        self.qkv = nn.Linear(dim, dim * self.qkv_scale//self.dim_reduction, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim//self.dim_reduction, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.dep_conv = nn.Conv2d(dim//self.dim_reduction//self.num_heads, self.ka*self.ka*dim//self.dim_reduction//self.num_heads, kernel_size=self.ka, bias=True, groups=dim//self.dim_reduction//self.num_heads, padding=self.ka//2, padding_mode=padding_mode)
        self.dep_conv1 = nn.Conv2d(dim//self.dim_reduction//self.num_heads, self.ka*self.ka*dim//self.dim_reduction//self.num_heads, kernel_size=self.ka, bias=True, groups=dim//self.dim_reduction//self.num_heads, padding=self.ka//2, padding_mode=padding_mode)
        if not share_dwc_kernel:
            self.dep_conv2 = nn.Conv2d(dim//self.dim_reduction//self.num_heads, self.ka*self.ka*dim//self.dim_reduction//self.num_heads, kernel_size=self.ka, bias=True, groups=dim//self.dim_reduction//self.num_heads, padding=self.ka//2, padding_mode=padding_mode)

        self.reset_parameters()

        # define a parameter table of relative position bias
        if self.rpb:
            self.relative_position_bias_table = nn.Parameter(
                torch.zeros(1, self.num_heads, 1, self.ka*self.ka, 1, 1))
            trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=3)

    def reset_parameters(self):
        """
        重置深度可分离卷积层的参数。

        为深度可分离卷积层的权重设置初始值，采用移位初始化的方式。
        """
        # shift initialization for group convolution
        kernel = torch.zeros(self.ka*self.ka, self.ka, self.ka)
        for i in range(self.ka*self.ka):
            kernel[i, i//self.ka, i%self.ka] = 1.
        kernel = kernel.unsqueeze(1).repeat(self.dim//self.dim_reduction//self.num_heads, 1, 1, 1)
        self.dep_conv.weight = nn.Parameter(data=kernel, requires_grad=False)

    def forward(self, x):
        """
        前向传播函数。

        参数:
        x (torch.Tensor): 输入的特征张量，形状为 (B, C, H, W)。

        返回:
        tuple: 包含三个元素，第一个元素是经过滑动注意力计算后的输出张量，形状为 (B, C, H, W)，
               后两个元素当前为 None。
        """
        x = einops.rearrange(x, 'b c h w -> b h w c')
        B, H, W, C = x.shape
        qkv = self.qkv(x)

        f_conv = qkv.permute(0, 3, 1, 2).reshape(B*self.num_heads, self.qkv_scale*C//self.dim_reduction//self.num_heads, H, W)

        if self.qkv_scale == 3:
            q = (f_conv[:, :C//self.dim_reduction//self.num_heads, :, :] * self.scale).reshape(B, self.num_heads, C//self.dim_reduction//self.num_heads, 1, H, W)
            k = f_conv[:, C//self.dim_reduction//self.num_heads:2*C//self.dim_reduction//self.num_heads, :, :] # B*self.nhead, C//self.nhead, H, W
            v = f_conv[:, 2*C//self.dim_reduction//self.num_heads:, :, :]
        elif self.qkv_scale == 1:
            q = (f_conv * self.scale).reshape(B, self.num_heads, C//self.dim_reduction//self.num_heads, 1, H, W)
            k = v = f_conv

        if self.share_dwc_kernel:
            k = (self.dep_conv(k) + self.dep_conv1(k)).reshape(B, self.num_heads, C//self.dim_reduction//self.num_heads, self.ka*self.ka, H, W)
            v = (self.dep_conv(v) + self.dep_conv1(v)).reshape(B, self.num_heads, C//self.dim_reduction//self.num_heads, self.ka*self.ka, H, W)
        else:
            k = (self.dep_conv(k) + self.dep_conv1(k)).reshape(B, self.num_heads, C//self.dim_reduction//self.num_heads, self.ka*self.ka, H, W)
            v = (self.dep_conv(v) + self.dep_conv2(v)).reshape(B, self.num_heads, C//self.dim_reduction//self.num_heads, self.ka*self.ka, H, W)

        if self.rpb:
            k = k + self.relative_position_bias_table
        attn = (q * k).sum(2, keepdim=True) # B, self.nhead, 1, k^2, H, W

        attn = self.softmax(attn)
        attn = self.attn_drop(attn)

        x = (attn * v).sum(3).reshape(B, C//self.dim_reduction, H, W).permute(0, 2, 3, 1)
        x = self.proj(x)
        x = self.proj_drop(x)
        x = einops.rearrange(x, 'b h w c -> b c h w')
        return x, None, None