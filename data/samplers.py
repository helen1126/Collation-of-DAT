# --------------------------------------------------------
# Swin Transformer
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu
# --------------------------------------------------------
# Vision Transformer with Deformable Attention
# Modified by Zhuofan Xia 
# --------------------------------------------------------

import torch

class SubsetRandomSampler(torch.utils.data.Sampler):
    r"""Samples elements randomly from a given list of indices, without replacement.

    Arguments:
        indices (sequence): a sequence of indices
    """

    def __init__(self, indices):
        """
        初始化 SubsetRandomSampler 类的实例。

        参数:
        indices (sequence): 一个索引序列，用于指定从数据集中采样的元素范围。
        """
        self.epoch = 0
        self.indices = indices

    def __iter__(self):
        """
        实现迭代器协议，用于随机迭代采样的索引。

        返回:
        generator: 一个生成器，用于生成随机排列的索引。
        """
        return (self.indices[i] for i in torch.randperm(len(self.indices)))

    def __len__(self):
        """
        返回采样器的长度，即采样的元素数量。

        返回:
        int: 采样器的长度，等于传入的索引序列的长度。
        """
        return len(self.indices)

    def set_epoch(self, epoch):
        """
        设置当前的训练轮数（epoch）。

        这个方法通常用于在分布式训练中同步不同进程的随机种子，
        以确保每个 epoch 的采样顺序是一致的。

        参数:
        epoch (int): 当前的训练轮数。
        """
        self.epoch = epoch
