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
        ��ʼ�� SubsetRandomSampler ���ʵ����

        ����:
        indices (sequence): һ���������У�����ָ�������ݼ��в�����Ԫ�ط�Χ��
        """
        self.epoch = 0
        self.indices = indices

    def __iter__(self):
        """
        ʵ�ֵ�����Э�飬�����������������������

        ����:
        generator: һ������������������������е�������
        """
        return (self.indices[i] for i in torch.randperm(len(self.indices)))

    def __len__(self):
        """
        ���ز������ĳ��ȣ���������Ԫ��������

        ����:
        int: �������ĳ��ȣ����ڴ�����������еĳ��ȡ�
        """
        return len(self.indices)

    def set_epoch(self, epoch):
        """
        ���õ�ǰ��ѵ��������epoch����

        �������ͨ�������ڷֲ�ʽѵ����ͬ����ͬ���̵�������ӣ�
        ��ȷ��ÿ�� epoch �Ĳ���˳����һ�µġ�

        ����:
        epoch (int): ��ǰ��ѵ��������
        """
        self.epoch = epoch
