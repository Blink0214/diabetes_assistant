import os

from experiments.expsimu import Expsimu


def adolescent():
    conv_kernel = [12, 16]
    decomp_kernel = []
    isometric_kernel = []
    seq_len = 96
    pred_len = 32
    # TODO
    for ii in conv_kernel:
        if ii % 2 == 0:  # the kernel of decomposition operation must be odd
            decomp_kernel.append(ii + 1)
            isometric_kernel.append((seq_len + pred_len + ii) // ii)
        else:
            decomp_kernel.append(ii)
            isometric_kernel.append((seq_len + pred_len + ii - 1) // ii)

    exp = Expsimu(seq_len=3361, pred_len=7, in_features=1, out_features=1, freq='min', mic_layers=1,
                  conv_kernel=conv_kernel, decomp_kernel=decomp_kernel, isometric_kernel=isometric_kernel,
                  seed=2024, subject='adolescent', dataset_dir=os.path.join('.', 'datasets', 'dl-TrainSet'))
    exp.train()
