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
        pe = torch.zeros(max_len, num_embed).float() # 初始化一个形状为 (max_len, num_embed) 的全零张量 pe，数据类型为浮点型（float）。
        pe.require_grad = False # 在训练过程中不需要计算该张量的梯度，即它是一个固定的常量。

        position = torch.arange(0, max_len).float().unsqueeze(1) 
        # 创建一个从 0 到 max_len-1 的张量 position，形状为 (max_len,)，然后使用 unsqueeze(1) 将其变为 (max_len, 1)，使其可以与后续的张量进行广播操作。
        div_term = (torch.arange(0, num_embed, 2).float() * -(math.log(10000.0) / num_embed)).exp()
        # 创建一个从 0 到 num_embed-1，步长为 2 的张量 div_term，形状为 (num_embed/2,)，然后对该张量进行如下变换：
        # 每个元素乘以 -(math.log(10000.0) / num_embed)。计算位置编码公式里的常量。
        # 对变换后的每个元素进行指数（exp()）运算。取指数。
        # 生成的 div_term 用于缩放 position 以得到不同频率的正弦和余弦波。

        pe[:, 0::2] = torch.sin(position * div_term) # pe[:,0::2] 从第 0 列开始，每隔两列选取一个元素
        pe[:, 1::2] = torch.cos(position * div_term)
        # 对 pe 的所有行的偶数列（即第 0, 2, 4, ... 列）赋值为 torch.sin(position * div_term) 计算的结果。
        # 对 pe 的所有行的奇数列（即第 1, 3, 5, ... 列）赋值为 torch.cos(position * div_term) 计算的结果。
        # 这样可以确保位置编码中包含正弦和余弦波，且这些波的频率随嵌入维度和位置的变化而变化。

        pe = pe.unsqueeze(0) # 将 pe 的第一个维度增加 1，变为 (1, max_len, num_embed)。这样做是为了便于在批处理时进行广播操作。
        self.register_buffer('pe', pe) # 将 pe 注册为模型的缓冲区（buffer），缓冲区是模型的状态的一部分，但不被视为模型参数，因此不会在训练过程中更新。这使得位置编码成为模型的一部分，并且在保存和加载模型时会被保留。

    def forward(self, x):
        # 在前向传播中，根据输入序列的长度截取相应长度的位置编码。
        return self.pe[:, :x.size(1)]


class TokenEmbedding(nn.Module):
    '''
    该模块用于将输入序列的每个 token（或特征）转换为嵌入向量。
    使用一个卷积层进行转换，将输入的通道数（in_channels）转换为指定数量的嵌入向量（num_embed）。

    对输入序列进行一维卷积操作，将其映射到嵌入空间。卷积层的权重使用 Kaiming 正态分布初始化，以适应 Leaky ReLU 激活函数。
    输入在进行卷积操作前后进行了维度变换，以确保卷积操作的正确性和输出形状的合理性。
    '''
    def __init__(self, in_channels, num_embed):
        # 在初始化时，采用 Kaiming 正态分布初始化权重。
        super(TokenEmbedding, self).__init__()
        padding = 1 if torch.__version__ >= '1.5.0' else 2
        self.tokenConv = nn.Conv1d(in_channels=in_channels, out_channels=num_embed,
                                   kernel_size=3, padding=padding, padding_mode='circular') # 一维卷积层 tokenConv
        # in_channels：输入通道数。
        # out_channels：输出通道数，即嵌入维度。
        # kernel_size=3：卷积核大小为 3。
        # padding=padding：根据前面计算的 padding 参数设置填充。
        '''在卷积操作中指定的零填充（zero-padding）的大小。填充是在输入序列的两端添加额外的值（通常是0），
        以使输入序列的长度在执行卷积操作时保持不变。
        填充有助于控制卷积层的输出大小，以便与输入大小相匹配。'''
        # padding_mode='circular'：使用循环填充模式，即边界元素填充到对侧。

        # 遍历模型中的所有模块，如果模块是 nn.Conv1d 类型，则使用 Kaiming 正态分布（He 初始化）初始化该卷积层的权重。
        # mode='fan_in' 和 nonlinearity='leaky_relu' 参数配置适用于带有 Leaky ReLU 激活函数的卷积层。
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')

    def forward(self, x):
        '''x.permute(0, 2, 1)：对输入 x 进行维度换位，将张量的第二和第三个维度互换。
        假设输入 x 的形状是 (batch_size, sequence_length, in_channels)，
        经过 permute 操作后形状变为 (batch_size, in_channels, sequence_length)。
        这是因为 nn.Conv1d 期望输入的形状是 (batch_size, in_channels, sequence_length)。
        self.tokenConv(...)：将换位后的张量输入到 tokenConv 卷积层中，进行卷积操作。
        .transpose(1, 2)：再次交换卷积结果的第二和第三个维度，
        使输出形状变为 (batch_size, sequence_length, num_embed)，与原始输入形状一致，只是通道数变为嵌入维度。
        return x：返回处理后的张量 x。'''
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
