# --------------------------------------------------------
# Swin Transformer
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu
# --------------------------------------------------------
# Vision Transformer with Deformable Attention
# Modified by Zhuofan Xia 
# --------------------------------------------------------

from .dat import DAT

def build_model(config):
    """
    根据给定的配置对象构建相应的模型。

    此函数接收一个配置对象，从中提取模型类型信息，并根据模型类型构建具体的模型实例。
    目前仅支持构建 'dat' 类型的模型。

    参数:
    config (CfgNode): 包含模型配置信息的对象，通常包含模型类型以及其他相关参数。

    返回:
    nn.Module: 根据配置构建的模型实例。

    异常:
    NotImplementedError: 如果配置中指定的模型类型不是 'dat'，则抛出此异常。
    """
    model_type = config.MODEL.TYPE
    if model_type == 'dat':
        model = DAT(**config.MODEL.DAT)
    else:
        raise NotImplementedError(f"Unkown model: {model_type}")

    return model
