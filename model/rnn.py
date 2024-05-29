import torch.nn as nn

class SimpleRNN(nn.Module):
    def __init__(self, input_size, embed_size, num_classes):
        super(SimpleRNN, self).__init__()
        self.rnn = nn.RNN(input_size, embed_size, batch_first=True) #　创建了一个RNN层，其中input_size是输入数据的特征维度，embed_size是RNN的隐藏状态大小，batch_first=True表示输入数据的维度顺序为(batch_size, seq_length, input_size)。
        self.fc = nn.Linear(embed_size, num_classes) # 创建了一个全连接（线性）层，将RNN的输出（最后一个时间步的隐藏状态）映射到类别空间，输出维度为num_classes，表示类别数量。

    def forward(self, x):
        out, _ = self.rnn(x) # 将输入数据x通过RNN层，得到输出out。在这里，_表示RNN的隐藏状态，因为在这个简单的模型中，我们不需要显式地处理隐藏状态。
        out = self.fc(out[:, -1, :]) # 取RNN输出序列的最后一个时间步的隐藏状态，然后通过全连接层self.fc进行分类，得到最终的输出。
        return out