import os

from experiments.expsimu import Expsimu
from experiments.expknn import Expknn

# 三个函数代表三个不同的实验

'''
name: 用于设置实验的名称，默认为 "simu"。
num_embed: 表示嵌入的维度大小。
num_hidden: 表示隐藏层的数量。
learning_rate: 学习率。
seq_len: 序列的长度。
in_features: 输入特征的数量。
freq: 数据的频率。
classes: 类别的数量。
lgf_layers: Luminous Group Fusion 的层数。
conv_kernel: 卷积核的大小。
isometric_kernel: 等距核的大小。
seed: 随机种子。
subject: 主题。
dataset_dir: 数据集所在的目录。
'''
def adolescent():
    conv_kernel = [48, 32, 24]
    isometric_kernel = []
    # seq_len = 480 # 为什么是480？
    seq_len = 96 # 一天的数据个数

    # TODO
    # 确保卷积核的大小在进行分解操作时是奇数。在卷积操作中，通常使用奇数大小的卷积核可以确保卷积操作的中心位置是存在的，从而更好地捕捉到数据的局部信息。
    # //是整数除法，取结果的整数部分A
    for ii in conv_kernel:
        if ii % 2 == 0:  # the kernel of decomposition operation must be odd
            isometric_kernel.append((seq_len + ii) // ii)
        else:
            isometric_kernel.append((seq_len + ii - 1) // ii)

    # exp = Expsimu(name='ykw', num_embed=64, num_hidden=8, learning_rate=0.01, seq_len=seq_len, in_features=1,
    #               freq='min', classes=11,
    #               lgf_layers=1, conv_kernel=conv_kernel, isometric_kernel=isometric_kernel,
    #               seed=2024, subject='adolescent', dataset_dir=os.path.join('.', 'datasets', 'dl-TrainSet'))

    # exp = Expsimu(name='ykw', num_embed=64, num_hidden=8, learning_rate=0.01, seq_len=seq_len, in_features=1,
    #               freq='min', classes=12,
    #               lgf_layers=1, conv_kernel=conv_kernel, isometric_kernel=isometric_kernel,
    #               seed=2024, subject='adolescent', dataset_dir=os.path.join('.', 'datasets', 'trainset'))

    # exp = Expsimu(name='cure', num_embed=64, num_hidden=8, learning_rate=0.01, seq_len=seq_len, in_features=1,
    #               freq='min', classes=35,
    #               lgf_layers=1, conv_kernel=conv_kernel, isometric_kernel=isometric_kernel,
    #               seed=2024, subject='adolescent', dataset_dir=os.path.join('.', 'datasets', 'trainset'))
    
    exp = Expsimu(name='adjust', num_embed=64, num_hidden=8, learning_rate=0.01, seq_len=seq_len, in_features=1,
                  freq='min', classes=93,
                  lgf_layers=1, conv_kernel=conv_kernel, isometric_kernel=isometric_kernel,
                  seed=2024, subject='adolescent', dataset_dir=os.path.join('.', 'datasets', 'trainset'))


    # exp.train()
    exp.test_knn()
    # exp.next_phase()


def rnn():
    exp = Expsimu(model='rnn', num_embed=32, learning_rate=0.01, classes=7, in_features=1,
                  seed=2024, subject='adolescent', dataset_dir=os.path.join('.', 'datasets', 'TrainSet'))
    exp.train()


def knn():
    exp = Expknn(name='knn', dataset_dir=os.path.join('.', 'datasets', 'TrainSet'))
    exp.test()
