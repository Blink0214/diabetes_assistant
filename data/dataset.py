import os.path
from itertools import accumulate

import numpy as np
import pandas as pd
import torch

from config.args import args
from torch.utils.data import Dataset
from utils.timefeature import time_features
from utils.tools import StandardScaler
import torch.nn.functional as F


class SIMU4AE(Dataset):
    def __init__(self, files, labels, mean=None, std=None, as_day=True):
        self.data = []
        self.time_stamp = []
        self.labels = labels
        self.as_day = as_day

        sample_count = []
        for path in files:
            df = pd.read_csv(path)
            df = df[:len(df) - (len(df) % args.seq_len)]

            if args.univariate:
                raw = df.iloc[:, [0, 1]]
                raw.columns = ['date', 'OT']
            else:
                # 'Date', 'CGM (mg / dl)', 'Dietary intake', 'Non-insulin hypoglycemic agents'
                # raw = df.iloc[:, [0, 1, 4, 7]]
                # raw.columns = ['date', 'OT', 'dietary_intake', 'non-insulin_hypoglycemic_agents']
                #
                # raw.loc[:, ['dietary_intake', 'non-insulin_hypoglycemic_agents']] = raw[
                #     ['dietary_intake', 'non-insulin_hypoglycemic_agents']].notna().astype(int)
                pass

            cols = list(raw.columns)
            cols.remove('OT')
            cols.remove('date')
            df_raw = raw[cols + ['OT']]
            df_raw['OT'].astype(float)
            time_stamp = time_features(raw, freq=args.freq)

            self.time_stamp.append(time_stamp)
            self.data.append(df_raw.values)
            sample_count.append((len(df_raw) - args.seq_len + 1) // (args.seq_len // 3))

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
        return self.cum_sum[-1] if self.as_day else len(self.data)

    def __getitem__(self, index):
        if self.as_day:
            fidx, offset = self._parse_index(index)
            offset = offset * (args.seq_len // 3)
            x_end = offset + args.seq_len
            return self.data[fidx][offset:x_end], self.time_stamp[fidx][offset:x_end], self.data[fidx][offset:x_end]
        else:
            label = torch.zeros((11,))
            label[self.labels[index]] = 1
            return self.data[index], self.time_stamp[index], label


class FinalData(Dataset):
    def __init__(self, start, end):
        final_data = pd.DataFrame(
            pd.read_csv(os.path.join(args.dataset_dir, 'final_data.csv'), encoding='ISO-8859-1')).replace('â\\x80?',
                                                                                                          '-')
        self.data = final_data.drop(labels='recommendation', axis=1).to_numpy()[start:end]
        self.labels = final_data.loc[:, 'recommendation'].to_numpy()[start:end]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index], self.labels[index]


class SIMU_old(Dataset):
    def __init__(self, files, labels, mean=None, std=None):
        self.data = []
        self.label = labels
        self.time_stamp = []

        for path in files:
            df = pd.read_csv(path)

            if args.univariate:
                raw = df.iloc[:, [0, 1]]
                raw.columns = ['date', 'OT']
            else:
                # 'Date', 'CGM (mg / dl)', 'Dietary intake', 'Non-insulin hypoglycemic agents'
                # raw = df.iloc[:, [0, 1, 4, 7]]
                # raw.columns = ['date', 'OT', 'dietary_intake', 'non-insulin_hypoglycemic_agents']
                #
                # raw.loc[:, ['dietary_intake', 'non-insulin_hypoglycemic_agents']] = raw[
                #     ['dietary_intake', 'non-insulin_hypoglycemic_agents']].notna().astype(int)
                pass

            cols = list(raw.columns)
            cols.remove('OT')
            cols.remove('date')
            df_raw = raw[cols + ['OT']]
            df_raw['OT'].astype(float)
            time_stamp = time_features(raw, freq=args.freq)

            self.time_stamp.append(time_stamp)
            self.data.append(df_raw.values)

        if mean is None and std is None:
            self.scaler = StandardScaler()
            self.scaler.fit(np.vstack(self.data))
            for x in range(len(self.data)):
                self.data[x] = self.scaler.transform(self.data[x])
        else:
            self.scaler = StandardScaler(mean=mean, std=std)
            for x in range(len(self.data)):
                self.data[x] = self.scaler.transform(self.data[x])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index], self.time_stamp[index], self.label[index]
