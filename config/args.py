import argparse
import logging as log

import torch

from config import setup_seed
import os

'''
--dropout: 模型中的dropout率，用于防止过拟合。
--freq: 时间特征编码的频率。
--in_features: 输入特征的数量。
--num_embed: 模型的嵌入向量的维度大小。
--univariate: 是否使用单变量模型。
--adjust_learning_rate: 是否调整学习率。
--batch_size: 训练数据的批大小。
--checkpoints: 模型检查点的保存位置。
--early_stop: 是否启用早停策略。
--dataset_dir: 数据集根目录。
--learning_rate: 优化器的学习率。
--patience: 早停策略的耐心程度，即在验证集上连续多少个epoch没有改进时停止训练。
--seed: 实验的随机种子，用于结果的可重复性。
--train_epochs: 训练的总epoch数。
--use_amp: 是否使用自动混合精度训练。
--gpu: 选择使用的GPU设备编号。
--use_multi_gpu: 是否使用多个GPU设备进行训练。
--devices: 多个GPU设备的ID。
'''

# 创建一个ArgumentParser对象，用于解析命令行参数。设置了一个描述信息，表示这是一个机器学习实验模板。
parser = argparse.ArgumentParser(description='[ML] Machine Learning Experiments Template')

# model
parser.add_argument('--dropout', type=float, default=0.05, help='input sequence length')
parser.add_argument('--freq', type=str, default="min", help='freq for time features encoding')
parser.add_argument('--in_features', type=int, default=7, help='input features')
parser.add_argument('--num_embed', type=int, default=512, help='dimension size of model')
parser.add_argument('--univariate', action='store_true', help='use multiple gpus', default=True) # action='store_true'表示如果命令行中出现了这个参数，则将其设为True；否则为False。

# experiments
parser.add_argument('--adjust_learning_rate', action='store_true', help='optimizer learning rate', default=True)
parser.add_argument('--batch_size', type=int, default=32, help='batch size of train data')
parser.add_argument('--checkpoints', type=str, default=os.path.join('.', 'checkpoints'),
                    help='location of model checkpoints') 
# 在当前工作目录下创建一个名为 "checkpoints" 的文件夹。
parser.add_argument('--early_stop', action='store_true', help='optimizer learning rate', default=True)
parser.add_argument('--dataset_dir', type=str, default=os.path.join('.', 'datasets'), help='datasets root directory')
parser.add_argument('--learning_rate', type=float, default=0.001, help='optimizer learning rate')
parser.add_argument('--patience', type=int, default=3, help='early stopping patience')
parser.add_argument('--seed', type=int, default=2024, help='seed of experiment')
parser.add_argument('--train_epochs', type=int, default=20, help='train epochs')
parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)
# parser.add_argument('--use_amp', help='use automatic mixed precision training', default=False)

parser.add_argument('--gpu', type=int, default=0, help='gpu device number')
parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
# parser.add_argument('--use_multi_gpu', help='use multiple gpus', default=False)
parser.add_argument('--devices', type=str, default='0,1,2,3', help='device ids of multiple gpus')

args = parser.parse_args()

setup_seed(args.seed) # 调用setup_seed函数，设置随机种子，以确保实验的可重复性。

# Check if a GPU is available
device = torch.device("cpu")
if torch.backends.mps.is_available():
    device = torch.device("mps")
    log.info("GPU Available. Using mps GPU.")
elif torch.cuda.is_available():
    device = torch.device(f"cuda:{args.gpu}")
    log.info("GPU Available. Using GPU[%d]: %s", args.gpu, torch.cuda.get_device_name(0))
else:
    log.info("No GPU available. Using CPU.")

if not os.path.exists(args.checkpoints):
    os.makedirs(args.checkpoints)
