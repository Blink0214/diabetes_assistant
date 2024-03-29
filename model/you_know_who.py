import torch
import torch.nn as nn

from model.embed import DataEmbedding
from model.local_global import MultiSeriesDecomp, LGF


class YKW(nn.Module):
    def __init__(self, in_features, seq_len,
                 decomp_kernel, conv_kernel, isometric_kernel,
                 out_features=1, num_hidden=64,
                 dropout=0.05, freq='min', device=torch.device('cuda:0')):
        super(YKW, self).__init__()

        self.seq_len = seq_len
        self.out_features = out_features
        self.decomp_kernel = decomp_kernel

        self.multi_moving_avg = MultiSeriesDecomp(decomp_kernel)
        self.embedding = DataEmbedding(in_features, num_hidden, freq, dropout)

        self.lgf = LGF(seq_len=seq_len, num_hidden=num_hidden, dropout=dropout,
                       decomp_kernel=decomp_kernel, conv_kernel=conv_kernel,
                       isometric_kernel=isometric_kernel, device=device)
        self.projection = nn.Linear(num_hidden, out_features)

        # self.regression = nn.Linear(seq_len, seq_len)
        # self.regression.weight = nn.Parameter((1 / seq_len) * torch.ones([seq_len, seq_len]), requires_grad=True)

    def forward(self, x, mark):
        # seq, trend = self.multi_moving_avg(x)
        # trend = self.regression(trend.permute(0, 2, 1)).permute(0, 2, 1)

        embedding = self.embedding(x, mark)

        fusion = self.lgf(embedding)
        fusion = self.projection(fusion)

        # out = fusion[:, -self.seq_len:, :] + trend[:, :, -self.out_features:]
        return fusion
