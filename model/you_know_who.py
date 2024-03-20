import torch
import torch.nn as nn

from model.embed import DataEmbedding
from model.local_global import MultiMovingAvg, Fusion


class YKW(nn.Module):
    def __init__(self, in_features, seq_len, classes,
                 num_hidden=512, lgf_layers=2,
                 dropout=0.05, freq='min',
                 device=torch.device('cuda:0'),
                 decomp_kernel=[33], conv_kernel=[12, 24], isometric_kernel=[18, 6], ):
        super(YKW, self).__init__()

        self.classes = classes
        self.seq_len = seq_len
        self.decomp_kernel = decomp_kernel

        self.multi_moving_avg = MultiMovingAvg(decomp_kernel)
        self.embedding = DataEmbedding(in_features, num_hidden, freq, dropout)

        self.local_global = Fusion(seq_len=seq_len, num_hidden=num_hidden,
                                   dropout=dropout,
                                   lgf_layers=lgf_layers, decomp_kernel=decomp_kernel,
                                   classes=classes,
                                   conv_kernel=conv_kernel,
                                   isometric_kernel=isometric_kernel, device=device)

    def forward(self, x, mark):
        seq = self.multi_moving_avg(x)[:, ::5, :][:, :self.seq_len, :]
        mark = mark[:, ::5, :][:, :self.seq_len, :]

        embedding = self.embedding(seq, mark)

        return self.local_global(embedding)
