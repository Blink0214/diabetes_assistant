import torch.nn as nn
import torch


class moving_avg(nn.Module):
    """
    Moving average block to highlight the trend of time series
    """

    def __init__(self, kernel_size, stride):
        super(moving_avg, self).__init__()
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


class series_decomp(nn.Module):
    """
    Series decomposition block
    """

    def __init__(self, kernel_size):
        super(series_decomp, self).__init__()
        self.moving_avg = moving_avg(kernel_size, stride=1)

    def forward(self, x):
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean


class series_decomp_multi(nn.Module):
    """
    Series decomposition block
    """

    def __init__(self, kernel_size):
        super(series_decomp_multi, self).__init__()
        self.kernel_size = kernel_size
        self.moving_avg = [moving_avg(kernel, stride=1) for kernel in kernel_size]

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


class MIC(nn.Module):
    """
    MIC layer to extract local and global features
    """

    def __init__(self, seq_len, pred_len, num_hidden=512, n_heads=8, dropout=0.05, decomp_kernel=[32], conv_kernel=[24],
                 isometric_kernel=[18, 6], device='cuda'):
        super(MIC, self).__init__()
        self.conv_kernel = conv_kernel
        self.isometric_kernel = isometric_kernel
        self.device = device

        # isometric convolution
        self.isometric_conv = nn.ModuleList([nn.Conv1d(in_channels=num_hidden, out_channels=num_hidden,
                                                       kernel_size=i, padding=0, stride=1)
                                             for i in isometric_kernel])

        # downsampling convolution: padding=i//2, stride=i
        self.conv = nn.ModuleList([nn.Conv1d(in_channels=num_hidden, out_channels=num_hidden,
                                             kernel_size=i, padding=i // 2, stride=i)
                                   for i in conv_kernel])

        # upsampling convolution
        self.conv_trans = nn.ModuleList([nn.ConvTranspose1d(in_channels=num_hidden, out_channels=num_hidden,
                                                            kernel_size=i, padding=0, stride=i)
                                         for i in conv_kernel])

        self.decomp = nn.ModuleList([series_decomp(k) for k in decomp_kernel])
        self.regression = nn.ModuleList([nn.Linear(seq_len + pred_len, seq_len + pred_len) for _ in decomp_kernel])
        for layer in self.regression:
            self.regression.weight = nn.Parameter(
                (1 / seq_len + pred_len) * torch.ones([seq_len + pred_len, seq_len + pred_len]), requires_grad=True)

        self.merge = torch.nn.Conv2d(in_channels=num_hidden, out_channels=num_hidden,
                                     kernel_size=(len(self.conv_kernel), 1))

        self.fnn = FeedForwardNetwork(num_hidden, num_hidden * 4, dropout)
        self.fnn_norm = torch.nn.LayerNorm(num_hidden)

        self.norm = torch.nn.LayerNorm(num_hidden)
        self.act = torch.nn.Tanh()
        self.drop = torch.nn.Dropout(dropout)

    def conv_trans_conv(self, input_, conv1d, conv1d_trans, isometric):
        batch, seq_len, channel = input_.shape
        x = input_.permute(0, 2, 1)

        # downsampling convolution
        x1 = self.drop(self.act(conv1d(x)))
        x = x1

        # isometric convolution
        zeros = torch.zeros((x.shape[0], x.shape[1], x.shape[2] - 1), device=self.device)
        x = torch.cat((zeros, x), dim=-1)
        x = self.drop(self.act(isometric(x)))
        x = self.norm((x + x1).permute(0, 2, 1)).permute(0, 2, 1)

        # upsampling convolution
        x = self.drop(self.act(conv1d_trans(x)))
        x = x[:, :, :seq_len]  # truncate

        x = self.norm(x.permute(0, 2, 1) + input_)
        return x

    def forward(self, src):
        # multi-scale
        multi = []
        for i in range(len(self.conv_kernel)):
            src_season, trend1 = self.decomp[i](src)
            trend = self.regression[i](trend1.permute(0, 2, 1)).permute(0, 2, 1)
            src_season = self.conv_trans_conv(src_season, self.conv[i], self.conv_trans[i], self.isometric_conv[i])
            multi.append(src_season + trend)

        # merge
        mg = torch.tensor([], device=self.device)
        for i in range(len(self.conv_kernel)):
            mg = torch.cat((mg, multi[i].unsqueeze(1)), dim=1)

        # merge input: batch_size, num_hidden, multi-scale(conv_kernel), seq_len
        mg = self.merge(mg.permute(0, 3, 1, 2)).squeeze(-2).permute(0, 2, 1)

        return self.fnn_norm(mg + self.fnn(mg))


class Seasonal_Prediction(nn.Module):
    def __init__(self, seq_len, pred_len, num_hidden, mic_layers, decomp_kernel, out_features,
                 conv_kernel, isometric_kernel, device, dropout=0.05):
        super(Seasonal_Prediction, self).__init__()

        self.mic = nn.ModuleList([MIC(seq_len=seq_len, pred_len=pred_len, num_hidden=num_hidden, dropout=dropout,
                                      decomp_kernel=decomp_kernel, conv_kernel=conv_kernel,
                                      isometric_kernel=isometric_kernel, device=device)
                                  for i in range(mic_layers)])

        self.projection = nn.Linear(num_hidden, out_features)

    def forward(self, x):
        for mic_layer in self.mic:
            x = mic_layer(x)
        return self.projection(x)
