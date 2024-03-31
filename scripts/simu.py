import os

from experiments.expsimu import Expsimu
from experiments.expknn import Expknn


def adolescent():
    conv_kernel = [48, 32, 24]
    isometric_kernel = []
    seq_len = 480
    # TODO
    for ii in conv_kernel:
        if ii % 2 == 0:  # the kernel of decomposition operation must be odd
            isometric_kernel.append((seq_len + ii) // ii)
        else:
            isometric_kernel.append((seq_len + ii - 1) // ii)

    exp = Expsimu(name='ykw', num_embed=64, num_hidden=8, learning_rate=0.01, seq_len=seq_len, in_features=1,
                  freq='min',
                  lgf_layers=1, conv_kernel=conv_kernel, isometric_kernel=isometric_kernel,
                  seed=2024, subject='adolescent', dataset_dir=os.path.join('.', 'datasets', 'TrainSet'))
    # exp.train()
    exp.test_knn()


def rnn():
    exp = Expsimu(model='rnn', num_embed=32, learning_rate=0.01, classes=7, in_features=1,
                  seed=2024, subject='adolescent', dataset_dir=os.path.join('.', 'datasets', 'TrainSet'))
    exp.train()


def knn():
    exp = Expknn(name='knn', dataset_dir=os.path.join('.', 'datasets', 'TrainSet'))
    exp.test()
