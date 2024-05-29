import torch
import torch.nn as nn

from model.embed import DataEmbedding
from model.local_global import MultiSeriesDecomp, LGF


'''
in_features：输入数据的特征维度。
seq_len：时间序列的长度。
conv_kernel 和 isometric_kernel：卷积核和等距核的大小。
out_features：模型的输出特征维度，默认为1。
num_embed 和 num_hidden：嵌入层和隐藏层的大小。
dropout：dropout的比例。
freq：时间频率。
device：模型计算的设备，默认为GPU。
'''

class YKW(nn.Module):
    '''
    定义了一个复杂的神经网络模型，用于处理时间序列数据，并在其基础上进行特征提取和预测任务。
    '''
    def __init__(self, in_features, seq_len,
                 conv_kernel, isometric_kernel,
                 out_features=1, num_embed=64, num_hidden=8,
                 dropout=0.05, freq='min', device=torch.device('cuda:0')):
        super(YKW, self).__init__()

        self.seq_len = seq_len
        self.out_features = out_features

        # 创建了一个数据嵌入层，将输入数据进行嵌入处理，以便模型更好地理解输入序列。
        self.embedding = DataEmbedding(in_features, num_embed, freq, dropout)

        # 创建了一个局部全局特征提取层，它将经过嵌入的序列数据进行处理，提取其局部和全局特征。
        self.lgf = LGF(seq_len=seq_len, num_embed=num_embed, num_hidden=num_hidden, dropout=dropout,
                       conv_kernel=conv_kernel, isometric_kernel=isometric_kernel, device=device)

        # 创建了一个线性投影层，将提取的特征映射到输出特征空间。
        self.projection = nn.Linear(num_embed, out_features)

    def forward(self, x, mark):
        embedding = self.embedding(x, mark)

        # print("embed形状：",embedding.shape)

        out = self.lgf(embedding)

        # print("lgf输出形状:",out.shape)
        # print("模型输出形状：",self.projection(out).shape)

        return self.projection(out)
