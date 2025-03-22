# --------------------------------------------------------
# Swin Transformer
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu
# --------------------------------------------------------
# Vision Transformer with Deformable Attention
# Modified by Zhuofan Xia 
# --------------------------------------------------------

import torch.optim as optim

def build_optimizer(config, model):
    """
    Build optimizer, set weight decay of normalization to 0 by default.
    """
    """
    根据配置文件和模型构建优化器，并默认将归一化层的权重衰减设置为0。

    该函数会先检查模型是否有指定不需要权重衰减的参数和关键字，
    以及是否有需要降低学习率的参数组。然后调用 set_weight_decay_and_lr 函数
    对模型参数进行分组，并根据配置文件中的优化器名称选择合适的优化器。

    参数:
    config (object): 配置对象，包含训练相关的配置参数，如优化器名称、学习率、权重衰减等。
    model (torch.nn.Module): 待训练的模型。

    返回:
    torch.optim.Optimizer: 构建好的优化器对象。
    """
    skip = {}
    skip_keywords = {}
    if hasattr(model, 'no_weight_decay'):
        skip = model.no_weight_decay()
    if hasattr(model, 'no_weight_decay_keywords'):
        skip_keywords = model.no_weight_decay_keywords()
    
    if hasattr(model, 'lower_lr_kvs'):
        lower_lr_kvs = model.lower_lr_kvs
    else:
        lower_lr_kvs = {}

    parameters = set_weight_decay_and_lr(
        model, skip, skip_keywords, lower_lr_kvs, config.TRAIN.BASE_LR)

    opt_lower = config.TRAIN.OPTIMIZER.NAME.lower()
    optimizer = None
    if opt_lower == 'sgd':
        optimizer = optim.SGD(parameters, momentum=config.TRAIN.OPTIMIZER.MOMENTUM, nesterov=True,
                              lr=config.TRAIN.BASE_LR, weight_decay=config.TRAIN.WEIGHT_DECAY)
    elif opt_lower == 'adamw':
        optimizer = optim.AdamW(parameters, eps=config.TRAIN.OPTIMIZER.EPS, betas=config.TRAIN.OPTIMIZER.BETAS,
                                lr=config.TRAIN.BASE_LR, weight_decay=config.TRAIN.WEIGHT_DECAY)
    
    return optimizer


def set_weight_decay_and_lr(
    model, 
    skip_list=(), skip_keywords=(), 
    lower_lr_kvs={}, base_lr=5e-4):
    """
    根据给定的规则对模型参数进行分组，并设置权重衰减和学习率。

    该函数会将模型参数分为需要权重衰减和不需要权重衰减的组，
    同时如果有需要降低学习率的参数组，会将其单独分组并设置较低的学习率。

    参数:
    model (torch.nn.Module): 待训练的模型。
    skip_list (tuple, 可选): 不需要权重衰减的参数名称列表，默认为空元组。
    skip_keywords (tuple, 可选): 包含不需要权重衰减的关键字的元组，默认为空元组。
    lower_lr_kvs (dict, 可选): 需要降低学习率的参数组及其对应的学习率缩放因子的字典，默认为空字典。
    base_lr (float, 可选): 基础学习率，默认为5e-4。

    返回:
    list: 包含参数分组信息的列表，每个元素是一个字典，指定了参数组和对应的权重衰减、学习率等。
    """
    # breakpoint()
    assert len(lower_lr_kvs) == 1 or len(lower_lr_kvs) == 0
    has_lower_lr = len(lower_lr_kvs) == 1
    if has_lower_lr:
        for k,v in lower_lr_kvs.items():
            lower_lr_key = k
            lower_lr = v * base_lr

    has_decay = []
    has_decay_low = []
    no_decay = []
    no_decay_low = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue  # frozen weights
        if len(param.shape) == 1 or name.endswith(".bias") or (name in skip_list) or \
                check_keywords_in_name(name, skip_keywords):

            if has_lower_lr and check_keywords_in_name(name, (lower_lr_key,)):
                no_decay_low.append(param)
            else:
                no_decay.append(param)
            
        else:

            if has_lower_lr and check_keywords_in_name(name, (lower_lr_key,)):
                has_decay_low.append(param)
            else:
                has_decay.append(param)

    if has_lower_lr:
        result = [
            {'params': has_decay},
            {'params': has_decay_low, 'lr': lower_lr},
            {'params': no_decay, 'weight_decay': 0.},
            {'params': no_decay_low, 'weight_decay': 0., 'lr': lower_lr}
        ]
    else:
        result = [
            {'params': has_decay},
            {'params': no_decay, 'weight_decay': 0.}
        ]
    # breakpoint()
    return result


def check_keywords_in_name(name, keywords=()):
    """
    检查给定的名称中是否包含指定的关键字。

    该函数会遍历关键字列表，检查每个关键字是否在名称中出现。

    参数:
    name (str): 待检查的名称。
    keywords (tuple, 可选): 关键字元组，默认为空元组。

    返回:
    bool: 如果名称中包含任何一个关键字，则返回 True；否则返回 False。
    """
    isin = False
    for keyword in keywords:
        if keyword in name:
            isin = True
    return isin
