import torch
import torch.nn as nn

class YKW(nn.Module):
    def __init__(self, in_features, out_features, seq_len, pred_len,
                 num_hidden=512, mic_layers=2,
                 dropout=0.05, embed='timeF', freq='min',
                 device=torch.device('cuda:0'), mode='regre',
                 decomp_kernel=[33], conv_kernel=[12, 24], isometric_kernel=[18, 6], ):
        super(YKW, self).__init__()


    def forward(self, x, mark):
        pass
