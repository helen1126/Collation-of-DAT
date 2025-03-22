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
import time
import argparse
import datetime
import numpy as np

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.distributed as dist
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from timm.utils import accuracy, AverageMeter

from data.config import get_config
from models import build_model
from data import build_loader
from utils.lr_scheduler import build_scheduler
from utils.optimizer import build_optimizer
from utils.logger import create_logger
from utils.utils import load_checkpoint, load_pretrained, save_checkpoint, \
                   get_grad_norm, auto_resume_helper, reduce_tensor

from torch.cuda.amp import GradScaler, autocast

import warnings
warnings.filterwarnings('ignore')


def parse_option():
    """
    解析命令行参数并加载配置文件。

    该函数创建一个命令行参数解析器，定义了一系列可选参数，
    包括配置文件路径、批量大小、数据集路径等。然后解析命令行参数，
    并调用 `get_config` 函数根据解析的参数加载配置。

    返回:
    tuple: 包含解析后的命令行参数和配置对象的元组。
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, required=True, metavar="FILE", help='path to config file', )
    parser.add_argument(
        "--opts",
        help="Modify config options by adding 'KEY VALUE' pairs. ",
        default=None,
        nargs='+',
    )
    # easy config modification
    parser.add_argument('--batch-size', type=int, help="batch size for single GPU")
    parser.add_argument('--data-path', type=str, help='path to dataset')
    parser.add_argument('--resume', help='resume from checkpoint')
    parser.add_argument('--amp', action='store_true', default=False)
    parser.add_argument('--output', default='output', type=str, metavar='PATH',
                        help='root of output folder, the full path is <output>/<model_name>/<tag> (default: output)')
    parser.add_argument('--tag', help='tag of experiment')
    parser.add_argument('--eval', action='store_true', help='Perform evaluation only')
    parser.add_argument('--throughput', action='store_true', help='Test throughput only')
    parser.add_argument('--print-freq', type=int, help='Printing frequency.', default=100)

    args, unparsed = parser.parse_known_args()

    config = get_config(args)

    return args, config


def main():
    """
    主函数，负责初始化训练环境、构建模型、数据加载器、优化器和学习率调度器，
    并根据配置进行模型训练、验证或吞吐量测试。

    该函数首先解析命令行参数和配置文件，初始化分布式训练环境，设置随机种子，
    然后根据配置调整学习率。接着创建日志记录器，保存配置文件，构建数据加载器和模型，
    初始化优化器和学习率调度器，选择合适的损失函数。根据配置加载预训练模型或恢复训练，
    最后根据配置进行训练、验证或吞吐量测试。
    """
    args, config = parse_option()
    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    world_size = int(os.environ['WORLD_SIZE'])
    dist.init_process_group(backend='nccl', init_method='env://', world_size=world_size, rank=rank) 
    torch.cuda.set_device(local_rank)
    dist.barrier()

    seed = config.SEED + dist.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    cudnn.enabled = True
    cudnn.benchmark = True

    if config.DATA.DATASET == 'imagenet':
        standard_bs = 512.0
    elif config.DATA.DATASET == 'imagenet22k':
        standard_bs = 4096.0
    else:
        raise RuntimeError("Wrong dataset!")

    # linear scale the learning rate according to total batch size, may not be optimal
    linear_scaled_lr = config.TRAIN.BASE_LR * config.DATA.BATCH_SIZE * dist.get_world_size() / standard_bs
    linear_scaled_warmup_lr = config.TRAIN.WARMUP_LR * config.DATA.BATCH_SIZE * dist.get_world_size() / standard_bs
    linear_scaled_min_lr = config.TRAIN.MIN_LR * config.DATA.BATCH_SIZE * dist.get_world_size() / standard_bs
    config.defrost()
    config.TRAIN.BASE_LR = linear_scaled_lr
    config.TRAIN.WARMUP_LR = linear_scaled_warmup_lr
    config.TRAIN.MIN_LR = linear_scaled_min_lr
    config.LOCAL_RANK = local_rank
    config.freeze()

    os.makedirs(config.OUTPUT, exist_ok=True)
    logger = create_logger(output_dir=config.OUTPUT, dist_rank=dist.get_rank(), name=f"{config.MODEL.NAME}")

    if dist.get_rank() == 0:
        path = os.path.join(config.OUTPUT, "config.yaml")
        with open(path, "w") as f:
            f.write(config.dump())
        logger.info(f"Full config saved to {path}")

    # print config
    logger.info(config.dump())
    
    _, dataset_val, data_loader_train, data_loader_val, mixup_fn = build_loader(config)

    logger.info(f"Creating model:{config.MODEL.TYPE}/{config.MODEL.NAME}")
    model = build_model(config)
    model.cuda()
    model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    logger.info(str(model))

    optimizer = build_optimizer(config, model)

    model = nn.parallel.DistributedDataParallel(
        model, device_ids=[local_rank], broadcast_buffers=True,
        find_unused_parameters=False)
    model_without_ddp = model.module
    
    lr_scheduler = build_scheduler(config, optimizer, len(data_loader_train))

    if config.AUG.MIXUP > 0.:
        # smoothing is handled with mixup label transform
        criterion = SoftTargetCrossEntropy()
    elif config.MODEL.LABEL_SMOOTHING > 0.:
        criterion = LabelSmoothingCrossEntropy(smoothing=config.MODEL.LABEL_SMOOTHING)
    else:
        criterion = nn.CrossEntropyLoss()

    max_accuracy = 0.0
    
    if config.MODEL.PRETRAINED is not None:
        pretrained_ckpt_path = config.MODEL.PRETRAINED
        load_pretrained(pretrained_ckpt_path, model_without_ddp, logger)

    if config.TRAIN.AUTO_RESUME and config.MODEL.RESUME == '':
        resume_file = auto_resume_helper(config.OUTPUT)
        if resume_file:
            if config.MODEL.RESUME:
                logger.warning(f"auto-resume changing resume file from {config.MODEL.RESUME} to {resume_file}")
            config.defrost()
            config.MODEL.RESUME = resume_file
            config.freeze()
            logger.info(f'auto resuming from {resume_file}')
        else:
            logger.info(f'no checkpoint found in {config.OUTPUT}, ignoring auto resume')

    if config.MODEL.RESUME:
        max_accuracy = load_checkpoint(config, model_without_ddp, optimizer, lr_scheduler, logger)
        acc1, acc5, loss = validate(config, data_loader_val, model, logger)
        torch.cuda.empty_cache()
        logger.info(f"Accuracy of the network on the {len(dataset_val)} test images: {acc1:.1f}%")
        if config.EVAL_MODE:
            return

    if config.THROUGHPUT_MODE:
        throughput(data_loader_val, model, logger)
        return
    logger.info("Start training")
    start_time = time.time()
    for epoch in range(config.TRAIN.START_EPOCH, config.TRAIN.EPOCHS):
        data_loader_train.sampler.set_epoch(epoch)

        train_one_epoch(config, model, criterion, data_loader_train, optimizer, epoch, mixup_fn, lr_scheduler, logger)

        if dist.get_rank() == 0 and ((epoch + 1) % config.SAVE_FREQ == 0 or (epoch + 1) == (config.TRAIN.EPOCHS)):
            save_checkpoint(config, epoch + 1, model_without_ddp, max_accuracy, optimizer, lr_scheduler, logger)

        acc1, acc5, loss = validate(config, data_loader_val, model, logger)

        logger.info(f"Accuracy of the network on the {len(dataset_val)} test images: {acc1:.1f}%")
        max_accuracy = max(max_accuracy, acc1)
        logger.info(f'Max accuracy: {max_accuracy:.2f}%')

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logger.info('Training time {}'.format(total_time_str))


def train_one_epoch(config, model, criterion, data_loader, optimizer, epoch, mixup_fn, lr_scheduler, logger):
    """
    训练一个 epoch 的函数。

    该函数将模型设置为训练模式，遍历训练数据加载器，对每个批次的数据进行前向传播、
    计算损失、反向传播和参数更新。根据配置选择是否使用自动混合精度训练，
    并根据配置进行梯度裁剪。同时记录训练过程中的损失、梯度范数和训练时间，
    并在一定间隔打印训练信息。

    参数:
    config (object): 配置对象，包含训练相关的配置信息。
    model (torch.nn.Module): 待训练的模型。
    criterion (torch.nn.Module): 损失函数。
    data_loader (torch.utils.data.DataLoader): 训练数据加载器。
    optimizer (torch.optim.Optimizer): 优化器。
    epoch (int): 当前训练的 epoch 数。
    mixup_fn (callable or None): 数据增强函数，如果为 None 则不使用。
    lr_scheduler (object): 学习率调度器。
    logger (object): 日志记录器。
    """
    model.train()
    optimizer.zero_grad()

    num_steps = len(data_loader)
    batch_time = AverageMeter()
    loss_meter = AverageMeter()
    norm_meter = AverageMeter()

    start = time.time()
    end = time.time()

    scaler = GradScaler()
    
    for idx, (samples, targets) in enumerate(data_loader):
        
        optimizer.zero_grad()
        samples = samples.cuda(non_blocking=True)
        targets = targets.cuda(non_blocking=True)

        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)
        
        if config.AMP: 
            with autocast():
                outputs, _, _ = model(samples)
                loss = criterion(outputs, targets)
            scaler.scale(loss).backward()
            if config.TRAIN.CLIP_GRAD:
                scaler.unscale_(optimizer)
                grad_norm = nn.utils.clip_grad_norm_(model.parameters(), config.TRAIN.CLIP_GRAD)
                scaler.step(optimizer)
                scaler.update()
            else:
                grad_norm = get_grad_norm(model.parameters())
                scaler.step(optimizer)
                scaler.update()
        else:
            outputs, _, _ = model(samples)
            loss = criterion(outputs, targets)
            loss.backward()
            if config.TRAIN.CLIP_GRAD:
                grad_norm = nn.utils.clip_grad_norm_(model.parameters(), config.TRAIN.CLIP_GRAD)
            else:
                grad_norm = get_grad_norm(model.parameters())
            optimizer.step()

        lr_scheduler.step_update(epoch * num_steps + idx)

        torch.cuda.synchronize()
        
        loss_meter.update(loss.item(), targets.size(0))
        norm_meter.update(grad_norm)
        batch_time.update(time.time() - end)
        end = time.time()

        if (idx + 1) % config.PRINT_FREQ == 0:
            lr = optimizer.param_groups[0]['lr']
            memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
            etas = batch_time.avg * (num_steps - idx)
            logger.info(
                f'Train: [{epoch + 1}/{config.TRAIN.EPOCHS}][{idx + 1}/{num_steps}]\t'
                f'eta {datetime.timedelta(seconds=int(etas))} lr {lr:.6f}\t'
                f'time {batch_time.val:.4f} ({batch_time.avg:.4f})\t'
                f'loss {loss_meter.val:.4f} ({loss_meter.avg:.4f})\t'
                f'grad_norm {norm_meter.val:.4f} ({norm_meter.avg:.4f})\t'
                f'mem {memory_used:.0f}MB')
    epoch_time = time.time() - start
    logger.info(f"EPOCH {epoch + 1} training takes {datetime.timedelta(seconds=int(epoch_time))}")

@torch.no_grad()
def validate(config, data_loader, model, logger):
    """
    在验证集上验证模型的函数。

    该函数将模型设置为评估模式，遍历验证数据加载器，对每个批次的数据进行前向传播，
    计算损失和准确率。使用 `reduce_tensor` 函数对损失和准确率进行规约操作，
    并记录验证过程中的损失、准确率和验证时间，最后打印验证结果。

    参数:
    config (object): 配置对象，包含验证相关的配置信息。
    data_loader (torch.utils.data.DataLoader): 验证数据加载器。
    model (torch.nn.Module): 待验证的模型。
    logger (object): 日志记录器。

    返回:
    tuple: 包含验证集上的 top-1 准确率、top-5 准确率和平均损失的元组。
    """
    criterion = nn.CrossEntropyLoss()
    model.eval()

    batch_time = AverageMeter()
    loss_meter = AverageMeter()
    acc1_meter = AverageMeter()
    acc5_meter = AverageMeter()

    end = time.time()
    for idx, (images, target) in enumerate(data_loader):
        images = images.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)

        # compute output
        output, _, _ = model(images)

        # measure accuracy and record loss
        loss = criterion(output, target)
        acc1, acc5 = accuracy(output, target, topk=(1, 5))

        acc1 = reduce_tensor(acc1)
        acc5 = reduce_tensor(acc5)
        loss = reduce_tensor(loss)

        loss_meter.update(loss.item(), target.size(0))
        acc1_meter.update(acc1.item(), target.size(0))
        acc5_meter.update(acc5.item(), target.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if (idx + 1) % config.PRINT_FREQ == 0:
            memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
            logger.info(
                f'Test: [{(idx + 1)}/{len(data_loader)}]\t'
                f'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                f'Loss {loss_meter.val:.4f} ({loss_meter.avg:.4f})\t'
                f'Acc@1 {acc1_meter.val:.3f} ({acc1_meter.avg:.3f})\t'
                f'Acc@5 {acc5_meter.val:.3f} ({acc5_meter.avg:.3f})\t'
                f'Mem {memory_used:.0f}MB')
    logger.info(f' * Acc@1 {acc1_meter.avg:.3f} Acc@5 {acc5_meter.avg:.3f}')
    return acc1_meter.avg, acc5_meter.avg, loss_meter.avg

@torch.no_grad()
def throughput(data_loader, model, logger):
    """
    测试模型吞吐量的函数。

    该函数将模型设置为评估模式，从数据加载器中获取一个批次的数据，
    先进行 50 次前向传播以热身，然后进行 30 次前向传播并记录时间，
    最后计算并打印模型的吞吐量。

    参数:
    data_loader (torch.utils.data.DataLoader): 数据加载器。
    model (torch.nn.Module): 待测试的模型。
    logger (object): 日志记录器。
    """
    model.eval()

    for _, (images, _) in enumerate(data_loader):
        images = images.cuda(non_blocking=True)
        batch_size = images.shape[0]
        for i in range(50):
            model(images)
        torch.cuda.synchronize()
        logger.info(f"throughput averaged with 30 times")
        tic1 = time.time()
        for i in range(30):
            model(images)
        torch.cuda.synchronize()
        tic2 = time.time()
        logger.info(f"batch_size {batch_size} throughput {30 * batch_size / (tic2 - tic1)}")
        return

if __name__ == '__main__':
    main()
