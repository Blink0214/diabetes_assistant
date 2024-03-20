import torch
import torch.nn as nn
import math


class PositionalEmbedding(nn.Module):
    def __init__(self, num_hidden, max_len=5000):
        super(PositionalEmbedding, self).__init__()
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, num_hidden).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, num_hidden, 2).float() * -(math.log(10000.0) / num_hidden)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)]


class TokenEmbedding(nn.Module):
    def __init__(self, in_channels, num_hidden):
        super(TokenEmbedding, self).__init__()
        padding = 1 if torch.__version__ >= '1.5.0' else 2
        self.tokenConv = nn.Conv1d(in_channels=in_channels, out_channels=num_hidden,
                                   kernel_size=3, padding=padding, padding_mode='circular')
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')

    def forward(self, x):
        x = self.tokenConv(x.permute(0, 2, 1)).transpose(1, 2)
        return x


class TimeFeatureEmbedding(nn.Module):
    def __init__(self, num_hidden, freq='h'):
        super(TimeFeatureEmbedding, self).__init__()

        freq_map = {'h': 4, 'min': 5, 's': 6, 'ME': 1, 'YE': 1, 'W': 2, 'D': 3, 'B': 3}
        in_feature = freq_map[freq]
        self.embed = nn.Linear(in_feature, num_hidden)

    def forward(self, x):
        return self.embed(x)


class DataEmbedding(nn.Module):
    def __init__(self, in_features, num_hidden, freq='h', dropout=0.1):
        super(DataEmbedding, self).__init__()

        self.value_embedding = TokenEmbedding(in_channels=in_features, num_hidden=num_hidden)
        self.position_embedding = PositionalEmbedding(num_hidden=num_hidden)
        self.temporal_embedding = TimeFeatureEmbedding(num_hidden=num_hidden, freq=freq)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, mark):
        x = self.value_embedding(x) + self.position_embedding(x) + self.temporal_embedding(mark)
        return self.dropout(x)
