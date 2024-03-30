import torch
import torch.nn as nn

from model.embed import DataEmbedding
from model.local_global import MultiSeriesDecomp, LGF


class YKW(nn.Module):
    def __init__(self, in_features, seq_len,
                 conv_kernel, isometric_kernel,
                 out_features=1, num_embed=64, num_hidden=8,
                 dropout=0.05, freq='min', device=torch.device('cuda:0')):
        super(YKW, self).__init__()

        self.seq_len = seq_len
        self.out_features = out_features

        self.embedding = DataEmbedding(in_features, num_embed, freq, dropout)

        self.lgf = LGF(seq_len=seq_len, num_embed=num_embed, num_hidden=num_hidden, dropout=dropout,
                       conv_kernel=conv_kernel, isometric_kernel=isometric_kernel, device=device)

    def forward(self, x, mark):
        embedding = self.embedding(x, mark)

        out = self.lgf(embedding)

        return out
