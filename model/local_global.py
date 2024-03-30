import torch.nn as nn
import torch


class MovingAvg(nn.Module):
    def __init__(self, kernel_size, stride=1):
        super(MovingAvg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        # x shape: batch,seq_len,channels
        # padding on the both ends of time series
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        return x


class SeriesDecomp(nn.Module):
    def __init__(self, kernel_size):
        super(SeriesDecomp, self).__init__()
        self.moving_avg = MovingAvg(kernel_size, stride=1)

    def forward(self, x):
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean


class MultiSeriesDecomp(nn.Module):
    def __init__(self, kernel_size):
        super(MultiSeriesDecomp, self).__init__()
        self.kernel_size = kernel_size
        self.moving_avg = [MovingAvg(kernel, stride=1) for kernel in kernel_size]

    def forward(self, x):
        moving_mean = []
        res = []
        for func in self.moving_avg:
            moving_avg = func(x)
            moving_mean.append(moving_avg)
            sea = x - moving_avg
            res.append(sea)

        sea = sum(res) / len(res)
        moving_mean = sum(moving_mean) / len(moving_mean)
        return sea, moving_mean


class FeedForwardNetwork(nn.Module):
    def __init__(self, in_feature, filter_size, dropout_rate=0.1):
        super(FeedForwardNetwork, self).__init__()

        self.layer1 = nn.Linear(in_feature, filter_size)
        self.relu = nn.ReLU()

        self.dropout = nn.Dropout(dropout_rate)
        self.layer2 = nn.Linear(filter_size, in_feature)

        self.initialize_weight(self.layer1)
        self.initialize_weight(self.layer2)

    def forward(self, x):
        x = self.layer1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.layer2(x)
        return x

    def initialize_weight(self, x):
        nn.init.xavier_uniform_(x.weight)
        if x.bias is not None:
            nn.init.constant_(x.bias, 0)


class Encoder(nn.Module):
    def __init__(self, seq_len, conv_kernel, isometric_kernel,
                 num_embed=512, dropout=0.05, device='cuda'):
        super(Encoder, self).__init__()
        # isometric convolution
        self.isometric_conv = nn.ModuleList([nn.Conv1d(in_channels=num_embed, out_channels=num_embed,
                                                       kernel_size=i, padding=0, stride=1)
                                             for i in isometric_kernel])

        # downsampling convolution: padding=i//2, stride=i
        self.conv = nn.ModuleList([nn.Conv1d(in_channels=num_embed, out_channels=num_embed,
                                             kernel_size=i, padding=i // 2, stride=i)
                                   for i in conv_kernel])

        self.merge = torch.nn.Conv2d(in_channels=num_embed, out_channels=num_embed,
                                     kernel_size=(len(self.conv_kernel), 1))

        self.norm = torch.nn.LayerNorm(num_embed)
        self.act = torch.nn.Tanh()
        self.drop = torch.nn.Dropout(dropout)

    def forward(self, src):
        # multi-scale
        multi = []
        for i in range(len(self.conv_kernel)):
            batch, seq_len, channel = src.shape
            x = src.permute(0, 2, 1)

            # downsampling convolution
            x1 = self.drop(self.act(self.conv_kernel[i](x)))
            x = x1

            # isometric convolution
            zeros = torch.zeros((x.shape[0], x.shape[1], x.shape[2] - 1), device=self.device)
            x = torch.cat((zeros, x), dim=-1)
            x = self.drop(self.act(self.isometric_conv[i](x)))
            x = self.norm((x + x1).permute(0, 2, 1)).permute(0, 2, 1)

            multi.append(x)

        # merge
        mg = torch.tensor([], device=self.device)
        for i in range(len(self.conv_kernel)):
            mg = torch.cat((mg, multi[i].unsqueeze(1)), dim=1)
        # merge input: batch_size, num_embed, multi-scale(conv_kernel), seq_len
        mg = self.merge(mg.permute(0, 3, 1, 2)).squeeze(-2).permute(0, 2, 1)
        return mg


class LGF(nn.Module):
    def __init__(self, seq_len, conv_kernel, isometric_kernel,
                 num_embed=512, dropout=0.05, device='cuda'):
        super(LGF, self).__init__()
        self.conv_kernel = conv_kernel
        self.isometric_kernel = isometric_kernel
        self.device = device

        self.encoder = Encoder(seq_len, conv_kernel, isometric_kernel, num_embed, dropout)

        # upsampling convolution
        self.decoder = nn.ModuleList([nn.ConvTranspose1d(in_channels=num_embed, out_channels=num_embed,
                                                            kernel_size=i, padding=0, stride=i)
                                         for i in conv_kernel])

        self.fnn = FeedForwardNetwork(num_embed, num_embed * 4, dropout)
        self.fnn_norm = torch.nn.LayerNorm(num_embed)

    def forward(self, src):
        fusion = self.encoder(src)
        out = self.decoder(fusion)

        return self.fnn_norm(out + self.fnn(out))

