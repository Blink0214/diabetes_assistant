import os
from experiments.expsht2 import ExpSHT2


def sht2():
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

    exp = ExpSHT2(seq_len=seq_len, pred_len=pred_len, in_features=1, out_features=1, freq='min', mic_layers=1,
                  conv_kernel=conv_kernel, decomp_kernel=decomp_kernel, isometric_kernel=isometric_kernel,
                  seed=2024, dataset_path=os.path.join('diabetes_datasets', 'Shanghai_T2DM'))
    exp.vali()
