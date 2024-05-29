import torch
import torch.nn as nn
import math

# 为神经网络提供一个丰富的输入表示，包括了数据值的向量化、位置信息的编码以及时间特征的嵌入，
class PositionalEmbedding(nn.Module):
    '''
    生成位置编码，以表示输入序列中元素的位置信息。位置编码是通过将正弦和余弦函数的值作为位置的函数而计算得出的，以保持相邻位置之间的关系。
    '''
    def __init__(self, num_embed, max_len=5000):
        # 在初始化时，计算了最大长度为 max_len 的位置编码。
        super(PositionalEmbedding, self).__init__()
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, num_embed).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, num_embed, 2).float() * -(math.log(10000.0) / num_embed)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # 在前向传播中，根据输入序列的长度截取相应长度的位置编码。
        return self.pe[:, :x.size(1)]


class TokenEmbedding(nn.Module):
    '''
    该模块用于将输入序列的每个 token（或特征）转换为嵌入向量。
    使用一个卷积层进行转换，将输入的通道数（in_channels）转换为指定数量的嵌入向量（num_embed）。
    '''
    def __init__(self, in_channels, num_embed):
        # 在初始化时，采用 Kaiming 正态分布初始化权重。
        super(TokenEmbedding, self).__init__()
        padding = 1 if torch.__version__ >= '1.5.0' else 2
        self.tokenConv = nn.Conv1d(in_channels=in_channels, out_channels=num_embed,
                                   kernel_size=3, padding=padding, padding_mode='circular')
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')

    def forward(self, x):
        x = self.tokenConv(x.permute(0, 2, 1)).transpose(1, 2)
        return x


class TimeFeatureEmbedding(nn.Module):
    '''
    该模块用于将时间特征转换为嵌入向量。它将时间的不同单位（例如小时、分钟、秒等）映射为不同的特征。
    '''
    def __init__(self, num_embed, freq='h'):
        # 在初始化时，采用线性层将时间特征映射为指定数量的嵌入向量。
        super(TimeFeatureEmbedding, self).__init__()

        freq_map = {'h': 4, 'min': 5, 's': 6, 'ME': 1, 'YE': 1, 'W': 2, 'D': 3, 'B': 3}
        in_feature = freq_map[freq]
        self.embed = nn.Linear(in_feature, num_embed)

    def forward(self, x):
        return self.embed(x)


class DataEmbedding(nn.Module):
    '''
    一个整合模块，将上述三种嵌入方式整合到一起。包含值嵌入（Value Embedding）、位置嵌入（Position Embedding）和时间嵌入（Time Embedding）。
    '''
    def __init__(self, in_features, num_embed, freq='h', dropout=0.1):
        super(DataEmbedding, self).__init__()

        self.value_embedding = TokenEmbedding(in_channels=in_features, num_embed=num_embed)
        self.position_embedding = PositionalEmbedding(num_embed=num_embed)
        self.temporal_embedding = TimeFeatureEmbedding(num_embed=num_embed, freq=freq)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, mark):
        # 在前向传播中，将输入数据和标记（mark）一起传入，并通过各个嵌入模块进行转换，然后将它们相加并应用 dropout。
        x = self.value_embedding(x) + self.position_embedding(x) + self.temporal_embedding(mark)
        return self.dropout(x)
