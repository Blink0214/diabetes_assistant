import torch
import torch.nn as nn
from config.args import args, device


'''
将输入的时间序列数据编码成特征向量，并基于这些特征向量进行分类预测。
'''
class CLS(nn.Module):
    def __init__(self, scale_size,
                 encoder,
                 # conv_kernel, isometric_kernel,
                 # out_features=1, num_embed=64, num_hidden=8,
                 # dropout=0.05, freq='min',
                 device=torch.device('cuda:0')):
        super(CLS, self).__init__()
        self.scale_size = scale_size # 缩放尺寸

        for param in encoder.parameters():
            param.requires_grad = False # 将编码器中所有参数的requires_grad属性设置为False。即在后续训练中，这些参数不会被计算梯度和更新，固定在当前数值状态。
        self.encoder = encoder.eval() # 编码器设置为评估模式。评估模式下，模型中的一些特定层（如Dropout层会表现不同，以确保模型评估时不引入随机性）

        self.rnn1 = nn.ModuleList([nn.RNN(1, 1, batch_first=True)
                                   for i in scale_size])
        '''
        创建了一个 ModuleList，其中包含了多个RNN模型。这些模型的数量由 scale_size 决定。
        每个RNN模型的输入大小为1，输出大小为1，且设置了 batch_first=True，表示输入数据的第一个维度是batch size。
        '''
        self.conv = nn.Conv2d(1, 1, kernel_size=(1, 3)) # 一个2D卷积层，输入通道数为1，输出通道数为1，卷积核的大小为(1, 3)。

        self.rnn2 = nn.RNN(1, 1, batch_first=True) # 另一个RNN模型，输入大小为1，输出大小为1，同样设置了 batch_first=True。

        self.norm = nn.BatchNorm1d(args.num_embed) # 一个Batch Normalization层，用于标准化输入数据。
        self.act = nn.ReLU() # ReLU激活函数，用于增加模型的非线性特性。

        self.fnn = nn.Linear(args.num_embed, args.classes) # 一个全连接层，将输入的特征向量映射到类别的数量上。
        self.fnn.weight = nn.Parameter((1 / args.classes) * torch.ones([args.classes, args.num_embed]),
                                       requires_grad=True) # 设置了全连接层的权重参数，初始化为均匀分布的值，并且允许在训练过程中更新。
        self.softmax = nn.Softmax(dim=1) # softmax函数，用于计算类别的概率分布。

    def forward(self, x, mark):
        '''
        模型的前向传播函数。接受输入数据x和时间标记mark，并对输入数据进行预处理，然后通过编码器编码成特征向量。
        将特征向量分别输入到多个尺度的RNN模型中，并将各个尺度的输出结果拼接在一起。
        然后通过卷积层和另一个RNN模型进行特征融合，并最终通过全连接层和softmax函数输出分类结果。
        '''
        batch_size = x.shape[0]
        data = x.reshape((-1, args.seq_len, x.shape[2])).float().to(device)
        time_stamp = mark.reshape(
            (-1, args.seq_len, mark.shape[2])).float().to(device)

        with torch.no_grad():
            a = self.encoder.lgf.encoder(self.encoder.embedding(data, time_stamp)) # 调用编码器的嵌入层和编码器层对输入数据进行编码，得到特征向量a.
            multi_scale = [a[i].reshape(-1, self.scale_size[i], 1) for i in range(len(a))] # 将特征向量a分别按照尺度进行reshape，得到多个尺度的特征向量列表。
        rst = [] # 创建空列表以存储RNN模型的输出结果。
        for i, x in enumerate(multi_scale, start=0): # 对多个尺度的特征向量进行遍历
            out, _ = self.rnn1[i](x) # 对当前尺度的特征向量输入到对应的RNN模型中进行处理，得到输出结果。
            rst.append(out[:, -1, :].reshape(batch_size, -1, args.num_embed, 1)) # 将RNN模型的输出结果进行切片操作，然后reshape成指定形状，并加入到结果列表 rst 中。
        # shape: batch_size, days, num_embed, multi_scale
        c = torch.concat(rst, dim=-1) # 将多个尺度的结果拼接在一起，形成一个新的张量 c。
        fusion = self.conv(c.permute(0, 2, 1, 3).reshape(-1, 1, c.shape[1], c.shape[3])) # 通过卷积层对拼接后的结果进行特征融合。
        out, _ = self.rnn2(fusion.reshape(-1, fusion.shape[2], 1)) # 将特征融合后的结果输入到第二个RNN模型中进行处理。
        c = self.softmax(self.fnn(self.act(self.norm(out[:, -1, :].reshape(batch_size, -1))))) # 通过Batch Normalization层、ReLU激活函数、全连接层和softmax函数得到最终的分类结果 c。
        return c
    # c.permute()是对张量中的维度进行重新排列，变成（batch_size, num_embed, days, multi_scale）
