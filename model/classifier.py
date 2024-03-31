import torch
import torch.nn as nn


class CLS(nn.Module):
    def __init__(self, scale_size,
                 # seq_len,
                 # conv_kernel, isometric_kernel,
                 # out_features=1, num_embed=64, num_hidden=8,
                 # dropout=0.05, freq='min',
                 device=torch.device('cuda:0')):
        super(CLS, self).__init__()

        self.rnn1 = nn.ModuleList([nn.RNN(i, 1, batch_first=True)
                                   for i in scale_size])

    def forward(self, multi_scale):
        # input: batch_size, scale_size,
        rst = []
        for i, x in enumerate(multi_scale, start=0):
            out, _ = self.rnn1[i](x)
            rst.append(out[:, -1, :])


a = [
    torch.randn((1, 64, 11)),
    torch.randn((1, 64, 16)),
    torch.randn((1, 64, 21)),
]

model = CLS([11, 16, 21])
model(a)
