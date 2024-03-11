import torch
import torch.nn as nn

from model.embed import DataEmbedding
from model.local_global import series_decomp_multi, Seasonal_Prediction


class MICN(nn.Module):
    def __init__(self, in_features, out_features, seq_len, pred_len,
                 num_hidden=512, mic_layers=2,
                 dropout=0.05, embed='timeF', freq='min',
                 device=torch.device('cuda:0'), mode='regre',
                 decomp_kernel=[33], conv_kernel=[12, 24], isometric_kernel=[18, 6], ):
        super(MICN, self).__init__()

        self.pred_len = pred_len
        self.seq_len = seq_len
        self.out_features = out_features
        self.decomp_kernel = decomp_kernel
        self.mode = mode

        self.decomp_multi = series_decomp_multi(decomp_kernel)

        # embedding
        self.season_embedding = DataEmbedding(in_features, num_hidden, embed, freq, dropout)

        self.conv_trans = Seasonal_Prediction(seq_len=seq_len, pred_len=pred_len, num_hidden=num_hidden,
                                              dropout=dropout,
                                              mic_layers=mic_layers, decomp_kernel=decomp_kernel,
                                              out_features=out_features,
                                              conv_kernel=conv_kernel,
                                              isometric_kernel=isometric_kernel, device=device)

        self.regression = nn.Linear(seq_len, pred_len)
        self.regression.weight = nn.Parameter((1 / pred_len) * torch.ones([pred_len, seq_len]), requires_grad=True)

    def forward(self, x, mark):

        # trend-cyclical prediction block: regre or mean
        if self.mode == 'regre':
            seasonal_init, trend = self.decomp_multi(x)
            trend = self.regression(trend.permute(0, 2, 1)).permute(0, 2, 1)
        elif self.mode == 'mean':
            mean = torch.mean(x, dim=1).unsqueeze(1).repeat(1, self.pred_len, 1)
            seasonal_init, trend = self.decomp_multi(x)
            trend = torch.cat([trend[:, -self.seq_len:, :], mean], dim=1)

        # embedding
        zeros = torch.zeros([x.shape[0], self.pred_len, x.shape[2]], device=x.device)
        seasonal = torch.cat([seasonal_init, zeros], dim=1)
        embedding = self.season_embedding(seasonal, mark)

        mic_out = self.conv_trans(embedding)
        out = mic_out[:, -self.pred_len:, :] + trend[:, :, -self.out_features:]
        return out
