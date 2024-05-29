import torch.nn as nn
import torch

# 时间序列分析
class MovingAvg(nn.Module):
    '''
    实现了移动平均的操作。在时间序列数据中，通过对每个时间点周围一定窗口内的数据取平均值，
    可以平滑数据，去除噪音，从而更好地观察数据的整体趋势。
    '''
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
    '''
    将时间序列数据分解成趋势和季节两部分。它利用了MovingAvg类来计算移动平均，
    然后从原始数据中减去移动平均部分得到趋势部分。
    '''
    def __init__(self, kernel_size):
        super(SeriesDecomp, self).__init__()
        self.moving_avg = MovingAvg(kernel_size, stride=1)

    def forward(self, x):
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean


class MultiSeriesDecomp(nn.Module):
    '''
    这个类是SeriesDecomp的扩展，可以处理多个时间序列数据。
    它通过多次调用MovingAvg类来计算多个时间序列数据的移动平均，
    并将每个时间序列数据减去对应的移动平均，
    最后将所有处理后的数据求和取平均，得到趋势部分。
    '''
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
    '''
    实现了一个前馈神经网络。它包括了两个线性层和一个ReLU激活函数，用于将输入数据映射到一个新的特征空间。
    '''
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
    '''
    编码器，用于将原始时间序列数据编码成多尺度的特征表示。它包括了多个卷积层和一些标准化、激活和 dropout 操作，用于提取输入数据的特征。
    '''
    def __init__(self, seq_len, conv_kernel, isometric_kernel,
                 num_embed=512, num_hidden=32, dropout=0.05, device='cuda'):
        super(Encoder, self).__init__()
        self.device = device
        # isometric convolution
        self.isometric_conv = nn.ModuleList([nn.Conv1d(in_channels=num_embed, out_channels=num_embed,
                                                       kernel_size=i, padding=0, stride=1)
                                             for i in isometric_kernel])

        # downsampling convolution: padding=i//2, stride=i
        self.conv = nn.ModuleList([nn.Conv1d(in_channels=num_embed, out_channels=num_embed,
                                             kernel_size=i, padding=i // 2, stride=i)
                                   for i in conv_kernel])

        self.norm = torch.nn.LayerNorm(num_embed)
        self.act = torch.nn.Tanh()
        self.drop = torch.nn.Dropout(dropout)

    def forward(self, src):
        # multi-scale
        multi = []
        for i in range(len(self.conv)):
            batch, seq_len, channel = src.shape
            x = src.permute(0, 2, 1)

            # downsampling convolution
            x1 = self.drop(self.act(self.conv[i](x)))
            x = x1

            # isometric convolution
            zeros = torch.zeros((x.shape[0], x.shape[1], x.shape[2] - 1), device=self.device)
            x = torch.cat((zeros, x), dim=-1)
            x = self.drop(self.act(self.isometric_conv[i](x)))
            x = self.norm((x + x1).permute(0, 2, 1)).permute(0, 2, 1)

            multi.append(x)
        return multi


class LGF(nn.Module):
    '''
    这个类实现了一个时间序列的低维空间特征表示模型。
    它包括了一个编码器和一个解码器，通过编码器将原始时间序列数据编码成低维特征表示，
    然后通过解码器将低维特征表示解码成原始时间序列数据。
    '''
    def __init__(self, seq_len, conv_kernel, isometric_kernel,
                 num_embed=512, num_hidden=32, dropout=0.05, device='cuda'):
        super(LGF, self).__init__()
        self.device = device
        self.seq_len = seq_len

        # self.encoder = Encoder(seq_len, conv_kernel, isometric_kernel, num_embed, num_hidden, dropout)
        self.encoder = Encoder(seq_len, conv_kernel, isometric_kernel, num_embed, num_hidden, dropout, device)

        self.fnn = nn.ModuleList([nn.Linear(in_features=i, out_features=num_hidden)
                                  for i in isometric_kernel])

        # upsampling convolution
        self.conv_trans = nn.ModuleList([nn.ConvTranspose1d(in_channels=num_embed, out_channels=num_embed,
                                                            kernel_size=i, padding=0, stride=i)
                                         for i in conv_kernel])

        self.merge = torch.nn.Conv2d(in_channels=num_embed, out_channels=num_embed,
                                     kernel_size=(len(conv_kernel), 1))

        self.fnn = FeedForwardNetwork(num_embed, num_embed * 4, dropout)
        self.fnn_norm = torch.nn.LayerNorm(num_embed)

        self.norm = torch.nn.LayerNorm(num_embed)
        self.act = torch.nn.Tanh()
        self.drop = torch.nn.Dropout(dropout)

    def forward(self, src):
        fusion = self.encoder(src)

        multi = []
        for i in range(len(self.conv_trans)):
            # upsampling convolution
            x = self.drop(self.act(self.conv_trans[i](fusion[i])))
            x = x[:, :, :self.seq_len]  # truncate

            x = self.norm(x.permute(0, 2, 1) + src)
            multi.append(x)

        # merge
        mg = torch.tensor([], device=self.device)
        for i in multi:
            mg = torch.cat((mg, i.unsqueeze(1)), dim=1)
        # merge input: batch_size, num_hidden, multi-scale(conv_kernel), seq_len
        mg = self.merge(mg.permute(0, 3, 1, 2)).squeeze(-2).permute(0, 2, 1)

        return self.fnn_norm(mg + self.fnn(mg))
