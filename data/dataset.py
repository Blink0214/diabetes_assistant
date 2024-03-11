import numpy as np
import pandas as pd

from config.args import args
from torch.utils.data import Dataset
from utils.timefeature import time_features
from utils.tools import StandardScaler
from itertools import accumulate


class SHT2(Dataset):
    def __init__(self, files, mean=None, std=None):
        self.data = []
        self.time_stamp = []

        sample_count = []
        for path in files:
            df = pd.read_excel(path)
            # 'Date', 'CGM (mg / dl)', 'Dietary intake', 'Non-insulin hypoglycemic agents'
            # raw = df.iloc[:, [0, 1, 4, 7]]
            raw = df.iloc[:, [0, 1]]
            raw.columns = ['date', 'OT']
            # raw.columns = ['date', 'OT', 'dietary_intake', 'non-insulin_hypoglycemic_agents']
            #
            # raw.loc[:, ['dietary_intake', 'non-insulin_hypoglycemic_agents']] = raw[
            #     ['dietary_intake', 'non-insulin_hypoglycemic_agents']].notna().astype(int)

            cols = list(raw.columns)
            cols.remove('OT')
            cols.remove('date')
            df_raw = raw[cols + ['OT']]
            df_raw['OT'].astype(float)
            time_stamp = time_features(raw, freq=args.freq)

            self.time_stamp.append(time_stamp)
            self.data.append(df_raw.values)
            sample_count.append(len(df_raw) - args.seq_len - args.pred_len + 1)

        self.cum_sum = list(accumulate(sample_count))

        if mean is None and std is None:
            self.scaler = StandardScaler()
            self.scaler.fit(np.vstack(self.data))
            for x in range(len(self.data)):
                self.data[x] = self.scaler.transform(self.data[x])
        else:
            self.scaler = StandardScaler(mean=mean, std=std)
            for x in range(len(self.data)):
                self.data[x] = self.scaler.transform(self.data[x])

    def _parse_index(self, idx):
        fidx = next(i for i, cum_sum in enumerate(self.cum_sum, start=0) if cum_sum >= idx + 1)
        offset = idx - (self.cum_sum[fidx - 1] if fidx > 0 else 0)
        return fidx, offset

    def __len__(self):
        return self.cum_sum[-1]

    def __getitem__(self, index):
        fidx, offset = self._parse_index(index)
        l_start = offset + args.seq_len
        l_end = offset + args.seq_len + args.pred_len
        label = self.data[fidx][l_start:l_end]
        mark = self.time_stamp[fidx][offset:l_end]
        return (self.data[fidx][offset:l_start], mark), label
