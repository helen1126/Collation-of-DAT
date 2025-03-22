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
from timm.scheduler.cosine_lr import CosineLRScheduler
from timm.scheduler.step_lr import StepLRScheduler
from timm.scheduler.scheduler import Scheduler


def build_scheduler(config, optimizer, n_iter_per_epoch):
    """
    ���������ļ�����ѧϰ�ʵ�������

    �ú������������ļ��е�ѵ������������ѵ������������������ѧϰ��˥�����Եȣ���
    ����Ż�����ÿ�ֵ���������������Ӧ��ѧϰ�ʵ�������

    ����:
    config (object): ���ö��󣬰���ѵ����ص����ò�����
    optimizer (torch.optim.Optimizer): �Ż�������
    n_iter_per_epoch (int): ÿ��ѵ���ĵ���������

    ����:
    Scheduler: ѧϰ�ʵ���������
    """
    num_steps = int(config.TRAIN.EPOCHS * n_iter_per_epoch)
    warmup_steps = int(config.TRAIN.WARMUP_EPOCHS * n_iter_per_epoch)
    decay_steps = int(config.TRAIN.LR_SCHEDULER.DECAY_EPOCHS * n_iter_per_epoch)

    lr_scheduler = None
    if config.TRAIN.LR_SCHEDULER.NAME == 'cosine':
        lr_scheduler = CosineLRScheduler(
            optimizer,
            t_initial=num_steps,
            lr_min=config.TRAIN.MIN_LR,
            warmup_lr_init=config.TRAIN.WARMUP_LR,
            warmup_t=warmup_steps,
            cycle_limit=1,
            t_in_epochs=False,
        )
    elif config.TRAIN.LR_SCHEDULER.NAME == 'linear':
        lr_scheduler = LinearLRScheduler(
            optimizer,
            t_initial=num_steps,
            lr_min_rate=0.01,
            warmup_lr_init=config.TRAIN.WARMUP_LR,
            warmup_t=warmup_steps,
            t_in_epochs=False,
        )
    elif config.TRAIN.LR_SCHEDULER.NAME == 'step':
        lr_scheduler = StepLRScheduler(
            optimizer,
            decay_t=decay_steps,
            decay_rate=config.TRAIN.LR_SCHEDULER.DECAY_RATE,
            warmup_lr_init=config.TRAIN.WARMUP_LR,
            warmup_t=warmup_steps,
            t_in_epochs=False,
        )

    return lr_scheduler


class LinearLRScheduler(Scheduler):
    def __init__(self,
                 optimizer: torch.optim.Optimizer,
                 t_initial: int,
                 lr_min_rate: float,
                 warmup_t=0,
                 warmup_lr_init=0.,
                 t_in_epochs=True,
                 noise_range_t=None,
                 noise_pct=0.67,
                 noise_std=1.0,
                 noise_seed=42,
                 initialize=True,
                 ) -> None:
        """
        ��ʼ������ѧϰ�ʵ�������

        �õ��������ݸ����Ĳ�����ʵ�����Ե�ѧϰ��˥�����ԣ�ͬʱ֧������׶Ρ�

        ����:
        optimizer (torch.optim.Optimizer): �Ż�������
        t_initial (int): ��ѵ��������
        lr_min_rate (float): ��Сѧϰ������ڳ�ʼѧϰ�ʵı�����
        warmup_t (int, ��ѡ): ����׶εĲ�����Ĭ��Ϊ 0��
        warmup_lr_init (float, ��ѡ): ����׶εĳ�ʼѧϰ�ʣ�Ĭ��Ϊ 0��
        t_in_epochs (bool, ��ѡ): �Ƿ�������Ϊ��λ���㲽����Ĭ��Ϊ True��
        noise_range_t (object, ��ѡ): ������Χ��Ĭ��Ϊ None��
        noise_pct (float, ��ѡ): �����ٷֱȣ�Ĭ��Ϊ 0.67��
        noise_std (float, ��ѡ): ������׼�Ĭ��Ϊ 1.0��
        noise_seed (int, ��ѡ): �������ӣ�Ĭ��Ϊ 42��
        initialize (bool, ��ѡ): �Ƿ��ʼ����Ĭ��Ϊ True��
        """
        super().__init__(
            optimizer, param_group_field="lr",
            noise_range_t=noise_range_t, noise_pct=noise_pct, noise_std=noise_std, noise_seed=noise_seed,
            initialize=initialize)

        self.t_initial = t_initial
        self.lr_min_rate = lr_min_rate
        self.warmup_t = warmup_t
        self.warmup_lr_init = warmup_lr_init
        self.t_in_epochs = t_in_epochs
        if self.warmup_t:
            self.warmup_steps = [(v - warmup_lr_init) / self.warmup_t for v in self.base_values]
            super().update_groups(self.warmup_lr_init)
        else:
            self.warmup_steps = [1 for _ in self.base_values]

    def _get_lr(self, t):
        """
        ���ݵ�ǰ��������ѧϰ�ʡ�

        �÷������ݵ�ǰ�������������׶κ�����˥�����ԣ�����ÿ���������ѧϰ�ʡ�

        ����:
        t (int): ��ǰ������

        ����:
        list: ÿ���������ѧϰ���б�
        """
        if t < self.warmup_t:
            lrs = [self.warmup_lr_init + t * s for s in self.warmup_steps]
        else:
            t = t - self.warmup_t
            total_t = self.t_initial - self.warmup_t
            lrs = [v - ((v - v * self.lr_min_rate) * (t / total_t)) for v in self.base_values]
        return lrs

    def get_epoch_values(self, epoch: int):
        """
        ���ݵ�ǰ������ȡѧϰ�ʡ�

        ���������Ϊ��λ���㲽��������� _get_lr ��������ѧϰ�ʣ����򷵻� None��

        ����:
        epoch (int): ��ǰ������

        ����:
        list or None: ÿ���������ѧϰ���б�� None��
        """
        if self.t_in_epochs:
            return self._get_lr(epoch)
        else:
            return None

    def get_update_values(self, num_updates: int):
        """
        ���ݵ�ǰ���´�����ȡѧϰ�ʡ�

        �����������Ϊ��λ���㲽��������� _get_lr ��������ѧϰ�ʣ����򷵻� None��

        ����:
        num_updates (int): ��ǰ���´�����

        ����:
        list or None: ÿ���������ѧϰ���б�� None��
        """
        if not self.t_in_epochs:
            return self._get_lr(num_updates)
        else:
            return None
