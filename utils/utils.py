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
    ��ָ��·�����ؼ����ļ����ָ�ģ�͡��Ż�����ѧϰ�ʵ�������״̬��

    �ú�������������ļ��еļ���·������ģ�Ͳ��������ڷ�����ģʽ�£�
    �ָ��Ż�����ѧϰ�ʵ�������״̬��ѵ����ʼ������

    ����:
    config (object): ���ö��󣬰���ģ�ͼ���·��������ģʽ��ѵ����ʼ������������Ϣ��
    model (torch.nn.Module): ���ָ���ģ�͡�
    optimizer (torch.optim.Optimizer): ���ָ����Ż�����
    lr_scheduler (object): ���ָ���ѧϰ�ʵ�������
    logger (object): ��־��¼�������ڼ�¼���ع����е���Ϣ��

    ����:
    float: �����м�¼�����׼ȷ�ʡ�
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
    ��ָ��·������Ԥѵ��ģ�͵Ĳ�����

    �ú��������Ԥѵ�������ļ���������ģ�͵� `load_pretrained` �������ز�����

    ����:
    ckpt_path (str): Ԥѵ�������ļ���·����
    model (torch.nn.Module): ������Ԥѵ��������ģ�͡�
    logger (object): ��־��¼�������ڼ�¼���ع����е���Ϣ��
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
    ���浱ǰģ�͡��Ż�����ѧϰ�ʵ�������״̬�Լ����׼ȷ�ʺ�ѵ�������������ļ���

    �ú����Ὣģ�͡��Ż�����ѧϰ�ʵ�������״̬�ֵ䣬�Լ����׼ȷ�ʺ�ѵ������
    ���浽ָ�����Ŀ¼�µļ����ļ��С�

    ����:
    config (object): ���ö��󣬰������Ŀ¼��������Ϣ��
    epoch (int): ��ǰѵ����������
    model (torch.nn.Module): �������ģ�͡�
    max_accuracy (float): ĿǰΪֹ�����׼ȷ�ʡ�
    optimizer (torch.optim.Optimizer): ��������Ż�����
    lr_scheduler (object): �������ѧϰ�ʵ�������
    logger (object): ��־��¼�������ڼ�¼��������е���Ϣ��
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
    ��������������ݶȷ�����

    �ú�������˵�û���ݶȵĲ�����Ȼ��������в����ݶȵ�ָ�����ͷ�����

    ����:
    parameters (torch.Tensor or list): �������ݶȷ����Ĳ����������ǵ��������������б�
    norm_type (float, ��ѡ): ���������ͣ�Ĭ��Ϊ 2����ʾ L2 ������

    ����:
    float: ���в����ݶȵ�ָ�����ͷ�����
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
    �Զ�����ָ�����Ŀ¼�µ����¼����ļ���

    �ú������г�ָ�����Ŀ¼�������� `.pth` ��β���ļ���
    �������ļ����޸�ʱ���ҵ����µļ����ļ���

    ����:
    output_dir (str): ���Ŀ¼��·����

    ����:
    str or None: ���¼����ļ���·�������û���ҵ��򷵻� None��
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
    �ڷֲ�ʽѵ�������У����������й�Լ��������Ͳ�ȡƽ������

    �ú����Ὣ��������������н��̼������ͣ�Ȼ����Խ��������õ�ƽ��ֵ��

    ����:
    tensor (torch.Tensor): ����Լ��������

    ����:
    torch.Tensor: ��Լ���������
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