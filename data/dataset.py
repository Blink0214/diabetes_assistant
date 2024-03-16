import numpy as np
import pandas as pd
import torch

from config.args import args
from torch.utils.data import Dataset
from utils.timefeature import time_features
from utils.tools import StandardScaler
import torch.nn.functional as F

class SIMU(Dataset):
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
