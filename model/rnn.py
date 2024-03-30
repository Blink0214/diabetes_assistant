import torch.nn as nn

class SimpleRNN(nn.Module):
    def __init__(self, input_size, embed_size, num_classes):
        super(SimpleRNN, self).__init__()
        self.rnn = nn.RNN(input_size, embed_size, batch_first=True)
        self.fc = nn.Linear(embed_size, num_classes)

    def forward(self, x):
        out, _ = self.rnn(x)
        out = self.fc(out[:, -1, :])
        return out