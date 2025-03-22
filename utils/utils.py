# --------------------------------------------------------
# Swin Transformer
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu
# --------------------------------------------------------
# Vision Transformer with Deformable Attention
# Modified by Zhuofan Xia 
# --------------------------------------------------------

import os
import torch
import torch.distributed as dist
import subprocess


def load_checkpoint(config, model, optimizer, lr_scheduler, logger):
    """
    从指定路径加载检查点文件，恢复模型、优化器和学习率调度器的状态。

    该函数会根据配置文件中的检查点路径加载模型参数，并在非评估模式下，
    恢复优化器、学习率调度器的状态和训练起始轮数。

    参数:
    config (object): 配置对象，包含模型检查点路径、评估模式、训练起始轮数等配置信息。
    model (torch.nn.Module): 待恢复的模型。
    optimizer (torch.optim.Optimizer): 待恢复的优化器。
    lr_scheduler (object): 待恢复的学习率调度器。
    logger (object): 日志记录器，用于记录加载过程中的信息。

    返回:
    float: 检查点中记录的最大准确率。
    """
    logger.info(f"==============> Resuming form {config.MODEL.RESUME}....................")
    if config.MODEL.RESUME.startswith('https'):
        checkpoint = torch.hub.load_state_dict_from_url(
            config.MODEL.RESUME, map_location='cpu', check_hash=True)
    else:
        checkpoint = torch.load(config.MODEL.RESUME, map_location='cpu')
    msg = model.load_state_dict(checkpoint['model'], strict=False)
    logger.info(msg)
    max_accuracy = 0.0
    if not config.EVAL_MODE and 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        config.defrost()
        config.TRAIN.START_EPOCH = checkpoint['epoch']
        config.freeze()
        logger.info(f"=> loaded successfully '{config.MODEL.RESUME}' (epoch {checkpoint['epoch']})")
        if 'max_accuracy' in checkpoint:
            max_accuracy = checkpoint['max_accuracy']

    del checkpoint
    torch.cuda.empty_cache()
    return max_accuracy

def load_pretrained(ckpt_path, model, logger):
    """
    从指定路径加载预训练模型的参数。

    该函数会加载预训练检查点文件，并调用模型的 `load_pretrained` 方法加载参数。

    参数:
    ckpt_path (str): 预训练检查点文件的路径。
    model (torch.nn.Module): 待加载预训练参数的模型。
    logger (object): 日志记录器，用于记录加载过程中的信息。
    """
    logger.info(f"==============> Loading pretrained form {ckpt_path}....................")
    checkpoint = torch.load(ckpt_path, map_location='cpu')
    msg = model.load_pretrained(checkpoint['model'])
    logger.info(msg)
    logger.info(f"=> Loaded successfully {ckpt_path} ")
    del checkpoint
    torch.cuda.empty_cache()


def save_checkpoint(config, epoch, model, max_accuracy, optimizer, lr_scheduler, logger):
    """
    保存当前模型、优化器、学习率调度器的状态以及最大准确率和训练轮数到检查点文件。

    该函数会将模型、优化器、学习率调度器的状态字典，以及最大准确率和训练轮数
    保存到指定输出目录下的检查点文件中。

    参数:
    config (object): 配置对象，包含输出目录等配置信息。
    epoch (int): 当前训练的轮数。
    model (torch.nn.Module): 待保存的模型。
    max_accuracy (float): 目前为止的最大准确率。
    optimizer (torch.optim.Optimizer): 待保存的优化器。
    lr_scheduler (object): 待保存的学习率调度器。
    logger (object): 日志记录器，用于记录保存过程中的信息。
    """
    save_state = {'model': model.state_dict(),
                  'optimizer': optimizer.state_dict(),
                  'lr_scheduler': lr_scheduler.state_dict(),
                  'max_accuracy': max_accuracy,
                  'epoch': epoch,
                  'config': config}

    save_path = os.path.join(config.OUTPUT, f'ckpt_epoch_{epoch}.pth')
    logger.info(f"{save_path} saving......")
    torch.save(save_state, save_path)
    logger.info(f"{save_path} saved !!!")


def get_grad_norm(parameters, norm_type=2):
    """
    计算给定参数的梯度范数。

    该函数会过滤掉没有梯度的参数，然后计算所有参数梯度的指定类型范数。

    参数:
    parameters (torch.Tensor or list): 待计算梯度范数的参数，可以是单个张量或张量列表。
    norm_type (float, 可选): 范数的类型，默认为 2，表示 L2 范数。

    返回:
    float: 所有参数梯度的指定类型范数。
    """
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = list(filter(lambda p: p.grad is not None, parameters))
    norm_type = float(norm_type)
    total_norm = 0
    for p in parameters:
        param_norm = p.grad.data.norm(norm_type)
        total_norm += param_norm.item() ** norm_type
    total_norm = total_norm ** (1. / norm_type)
    return total_norm


def auto_resume_helper(output_dir):
    """
    自动查找指定输出目录下的最新检查点文件。

    该函数会列出指定输出目录下所有以 `.pth` 结尾的文件，
    并根据文件的修改时间找到最新的检查点文件。

    参数:
    output_dir (str): 输出目录的路径。

    返回:
    str or None: 最新检查点文件的路径，如果没有找到则返回 None。
    """
    checkpoints = os.listdir(output_dir)
    checkpoints = [ckpt for ckpt in checkpoints if ckpt.endswith('pth')]
    print(f"All checkpoints founded in {output_dir}: {checkpoints}")
    if len(checkpoints) > 0:
        latest_checkpoint = max([os.path.join(output_dir, d) for d in checkpoints], key=os.path.getmtime)
        print(f"The latest checkpoint founded: {latest_checkpoint}")
        resume_file = latest_checkpoint
    else:
        resume_file = None
    return resume_file

def reduce_tensor(tensor):
    """
    在分布式训练环境中，对张量进行规约操作（求和并取平均）。

    该函数会将输入的张量在所有进程间进行求和，然后除以进程总数得到平均值。

    参数:
    tensor (torch.Tensor): 待规约的张量。

    返回:
    torch.Tensor: 规约后的张量。
    """
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= dist.get_world_size()
    return rt

def init_dist_slurm():
    """Initialize slurm distributed training environment.
    If argument ``port`` is not specified, then the master port will be system
    environment variable ``MASTER_PORT``. If ``MASTER_PORT`` is not in system
    environment variable, then a default port ``29500`` will be used.
    Args:
        backend (str): Backend of torch.distributed.
        port (int, optional): Master port. Defaults to None.
    """
    proc_id = int(os.environ['SLURM_PROCID'])
    ntasks = int(os.environ['SLURM_NTASKS'])
    node_list = os.environ['SLURM_NODELIST']
    num_gpus = torch.cuda.device_count()
    torch.cuda.set_device(proc_id % num_gpus)
    addr = subprocess.getoutput(
        f'scontrol show hostname {node_list} | head -n1')
    # specify master port
    if 'MASTER_PORT' in os.environ:
        pass  # use MASTER_PORT in the environment variable
    else:
        # 29500 is torch.distributed default port
        os.environ['MASTER_PORT'] = '29500'
    # use MASTER_ADDR in the environment variable if it already exists
    if 'MASTER_ADDR' not in os.environ:
        os.environ['MASTER_ADDR'] = addr
    
    os.environ['WORLD_SIZE'] = str(ntasks)
    os.environ['LOCAL_RANK'] = str(proc_id % num_gpus)
    os.environ['RANK'] = str(proc_id)

    dist.init_process_group(backend='nccl')