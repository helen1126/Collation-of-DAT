# Collation-of-DAT
This is the history code collation for the artical--Vision Transformer with Deformable Attention(https://arxiv.org/abs/2201.00520)

## Dependencies

- NVIDIA GPU + CUDA 11.3
- Python 3.9
- PyTorch == 1.11.0
- torchvision == 0.12.0
- numpy == 1.20.3
- timm == 0.5.4
- einops == 0.6.1
- natten == 0.14.6
- PyYAML
- yacs
- termcolor

## Evaluate Pretrained Models on ImageNet-1K Classification

We provide the pretrained models in the tiny, small, and base versions of DAT++, as listed below.

| model  | resolution | acc@1 | config | pretrained weights |
| :---: | :---: | :---: | :---: | :---: |
| DAT-T++ | 224x224 | 83.9 | [config](configs/dat_tiny.yaml) | [OneDrive](https://1drv.ms/u/s!ApI0vb6wPqmtgrl-pI8MPFoll-ueNQ?e=bpdieu) / [TsinghuaCloud](https://cloud.tsinghua.edu.cn/f/14c5ddae10b642e68089/) |
| DAT-S++ | 224x224 | 84.6 | [config](configs/dat_small.yaml) | [OneDrive](https://1drv.ms/u/s!ApI0vb6wPqmtgroB0ESeknbTsksWAg?e=Jbh0BS) / [TsinghuaCloud](https://cloud.tsinghua.edu.cn/f/4c2a76360c964fbd81d5/) |
| DAT-B++ | 224x224 | 84.9 | [config](configs/dat_base.yaml) | [OneDrive](https://1drv.ms/u/s!ApI0vb6wPqmtgrl_P46QOehhgA0-wg?e=DJRAfw) / [TsinghuaCloud](https://cloud.tsinghua.edu.cn/f/8e30492404d348d89f25/) |
| DAT-B++ | 384x384 | 85.9 | [config](configs/dat_base_384.yaml) | [OneDrive](https://1drv.ms/u/s!ApI0vb6wPqmtgroAI7cLAoj17khZNw?e=7yzxAg) / [TsinghuaCloud](https://cloud.tsinghua.edu.cn/f/032dc804cdf44bf18bb5/) |

To evaluate one model, please download the pretrained weights to your local machine and run the script `evaluate.sh` as follow. 

**Please notice: Before training or evaluation, please set the `--data-path` argument in `train.sh` or `evaluate.sh` to the path where ImageNet-1K data stores.**

```
bash evaluate.sh <gpu_nums> <path-to-config> <path-to-pretrained-weights>
```

E.g., suppose evaluating the DAT-Tiny model (`dat_pp_tiny_in1k_224.pth`) with 8 GPUs, the command should be:

```
bash evaluate.sh 8 configs/dat_tiny.yaml dat_pp_tiny_in1k_224.pth
```

And the evaluation result should give:

```
[2023-09-04 17:18:15 dat_plus_plus] (main.py 301): INFO  * Acc@1 83.864 Acc@5 96.734
[2023-09-04 17:18:15 dat_plus_plus] (main.py 179): INFO Accuracy of the network on the 50000 test images: 83.9%
```


## Train Models from Scratch

To train a model from scratch, we provide a simple script `train.sh`. E.g, to train a model with 8 GPUs on a single node, you can use this command:

```
bash train.sh 8 <path-to-config> <experiment-tag>
```

We also provide a training script `train_slurm.sh` for training models on multiple machines with a larger batch-size like 4096. 

```
bash train_slurm.sh 32 <path-to-config> <slurm-job-name>
```

**Remember to change the \<path-to-imagenet\> in the script files to your own ImageNet directory.**