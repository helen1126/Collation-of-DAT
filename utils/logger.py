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
import sys
import logging
import functools
from termcolor import colored

@functools.lru_cache()
def create_logger(output_dir, dist_rank=0, name=''):
    """
    ����һ����־��¼�������ڼ�¼�������й����е���Ϣ��

    �ú�������ݸ��������Ŀ¼���ֲ�ʽѵ������������־���ƴ���һ����־��¼����
    ���������̣�dist_rank Ϊ 0������־��Ϣ��ͬʱ���������̨���ļ��������������̣���־��Ϣ��������ļ���

    ����:
    output_dir (str): ��־�ļ������Ŀ¼��
    dist_rank (int, ��ѡ): �ֲ�ʽѵ���еĽ���������Ĭ��Ϊ 0��
    name (str, ��ѡ): ��־��¼�������ƣ�Ĭ��Ϊ���ַ�����

    ����:
    logging.Logger: �����õ���־��¼������
    """
    # create logger
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    # create formatter
    fmt = '[%(asctime)s %(name)s] (%(filename)s %(lineno)d): %(levelname)s %(message)s'
    color_fmt = colored('[%(asctime)s %(name)s]', 'green') + \
                colored('(%(filename)s %(lineno)d)', 'yellow') + ': %(levelname)s %(message)s'

    # create console handlers for master process
    if dist_rank == 0:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.DEBUG)
        console_handler.setFormatter(
            logging.Formatter(fmt=color_fmt, datefmt='%Y-%m-%d %H:%M:%S'))
        logger.addHandler(console_handler)
        file_handler = logging.FileHandler(os.path.join(output_dir, f'log_rank{dist_rank}.txt'), mode='a')
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(logging.Formatter(fmt=fmt, datefmt='%Y-%m-%d %H:%M:%S'))
        logger.addHandler(file_handler)

    return logger
