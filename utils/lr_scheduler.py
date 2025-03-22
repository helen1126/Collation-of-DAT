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
    根据配置文件构建学习率调度器。

    该函数根据配置文件中的训练参数（如总训练轮数、热身轮数、学习率衰减策略等），
    结合优化器和每轮迭代次数，构建相应的学习率调度器。

    参数:
    config (object): 配置对象，包含训练相关的配置参数。
    optimizer (torch.optim.Optimizer): 优化器对象。
    n_iter_per_epoch (int): 每轮训练的迭代次数。

    返回:
    Scheduler: 学习率调度器对象。
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
        初始化线性学习率调度器。

        该调度器根据给定的参数，实现线性的学习率衰减策略，同时支持热身阶段。

        参数:
        optimizer (torch.optim.Optimizer): 优化器对象。
        t_initial (int): 总训练步数。
        lr_min_rate (float): 最小学习率相对于初始学习率的比例。
        warmup_t (int, 可选): 热身阶段的步数，默认为 0。
        warmup_lr_init (float, 可选): 热身阶段的初始学习率，默认为 0。
        t_in_epochs (bool, 可选): 是否以轮数为单位计算步数，默认为 True。
        noise_range_t (object, 可选): 噪声范围，默认为 None。
        noise_pct (float, 可选): 噪声百分比，默认为 0.67。
        noise_std (float, 可选): 噪声标准差，默认为 1.0。
        noise_seed (int, 可选): 噪声种子，默认为 42。
        initialize (bool, 可选): 是否初始化，默认为 True。
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
        根据当前步数计算学习率。

        该方法根据当前步数，结合热身阶段和线性衰减策略，计算每个参数组的学习率。

        参数:
        t (int): 当前步数。

        返回:
        list: 每个参数组的学习率列表。
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
        根据当前轮数获取学习率。

        如果以轮数为单位计算步数，则调用 _get_lr 方法计算学习率；否则返回 None。

        参数:
        epoch (int): 当前轮数。

        返回:
        list or None: 每个参数组的学习率列表或 None。
        """
        if self.t_in_epochs:
            return self._get_lr(epoch)
        else:
            return None

    def get_update_values(self, num_updates: int):
        """
        根据当前更新次数获取学习率。

        如果不以轮数为单位计算步数，则调用 _get_lr 方法计算学习率；否则返回 None。

        参数:
        num_updates (int): 当前更新次数。

        返回:
        list or None: 每个参数组的学习率列表或 None。
        """
        if not self.t_in_epochs:
            return self._get_lr(num_updates)
        else:
            return None
