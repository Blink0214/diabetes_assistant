import torch
import torch.nn as nn
from config.args import args, device


class CLS(nn.Module):
    def __init__(self, scale_size,
                 encoder,
                 # conv_kernel, isometric_kernel,
                 # out_features=1, num_embed=64, num_hidden=8,
                 # dropout=0.05, freq='min',
                 device=torch.device('cuda:0')):
        super(CLS, self).__init__()
        self.scale_size = scale_size

        for param in encoder.parameters():
            param.requires_grad = False
        self.encoder = encoder.eval()

        self.rnn1 = nn.ModuleList([nn.RNN(1, 1, batch_first=True)
                                   for i in scale_size])
        self.conv = nn.Conv2d(1, 1, kernel_size=(1, 3))

        self.rnn2 = nn.RNN(1, 1, batch_first=True)

        self.norm = nn.BatchNorm1d(args.num_embed)
        self.act = nn.ReLU()

        self.fnn = nn.Linear(args.num_embed, args.classes)
        self.fnn.weight = nn.Parameter((1 / args.classes) * torch.ones([args.classes, args.num_embed]),
                                       requires_grad=True)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x, mark):
        data = x.reshape((-1, args.seq_len, x.shape[2])).float().to(device)
        time_stamp = mark.reshape(
            (-1, args.seq_len, mark.shape[2])).float().to(device)

        with torch.no_grad():
            a = self.encoder.lgf.encoder(self.encoder.embedding(data, time_stamp))
            multi_scale = [a[i].reshape(-1, self.scale_size[i], 1) for i in range(len(a))]
        rst = []
        for i, x in enumerate(multi_scale, start=0):
            out, _ = self.rnn1[i](x)
            rst.append(out[:, -1, :].reshape(args.batch_size, -1, args.num_embed, 1))
        # shape: batch_size, days, num_embed, multi_scale
        c = torch.concat(rst, dim=-1)
        fusion = self.conv(c.permute(0, 2, 1, 3).reshape(-1, 1, c.shape[1], c.shape[3]))
        out, _ = self.rnn2(fusion.reshape(-1, fusion.shape[2], 1))
        c = self.softmax(self.fnn(self.act(self.norm(out[:, -1, :].reshape(args.batch_size, -1)))))
        return c
