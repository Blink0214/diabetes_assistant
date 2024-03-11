import torch
import torch.nn as nn


class BaselPred(nn.Module):
    def __init__(self, seq_len, pred_len):
        super(BaselPred, self).__init__()

        self.seq_len = seq_len
        self.pred_len = pred_len

    def forward(self, time_feature, seq):
        pass
