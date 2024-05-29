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
    '''
    主要用于加载用于自动编码器 (Autoencoder) 训练的数据集。
    files: 文件路径列表，用于指定要加载的数据文件。
    labels: 标签列表，对应每个数据文件的标签。
    mean和std: 均值和标准差，用于标准化数据。默认为None。
    as_day: 布尔值，指示是否按天划分数据。默认为True。
    '''
    def __init__(self, files, labels, mean=None, std=None, as_day=True):
    # def __init__(self, files, labels, mean=None, std=None, as_day=False):
        self.data = [] # 用于存储加载的数据。
        self.time_stamp = [] # 用于存储时间戳特征。
        self.labels = labels # 存储标签。
        self.as_day = as_day # 标志位，指示数据是否按天划分。

        sample_count = []
        # 这段代码遍历文件路径列表，读取每个文件的CSV数据，并截断到能够完整容纳序列长度的位置。
        for index, path in enumerate(files) :
            df = pd.read_csv(path)
            df = df[:len(df) - (len(df) % args.seq_len)]

            # 这段代码根据配置参数args.univariate决定如何处理数据。如果为True，则选择前两列作为数据。否则，根据注释中的说明选择特定的列。
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

            # 重新组织DataFrame的列顺序，确保"OT"列在最后，并将"OT"列的数据类型转换为浮点型。
            cols = list(raw.columns)
            # print("文件列：",cols)
            cols.remove('OT')
            cols.remove('date')
            df_raw = raw[cols + ['OT']]
            # print("df_raw的文件列：",list(df_raw.columns))
            # print(df_raw.head())
            # print("文件长度：",len(df_raw))
            df_raw['OT'].astype(float)
            # print("raw的文件列：",list(raw.columns))
            # print(raw.head())
            time_stamp = time_features(raw, freq=args.freq)

            # 调用time_features函数生成时间戳特征，并将时间戳和数据存储到相应的列表中。同时，计算样本数量并存储到sample_count列表中。
            self.time_stamp.append(time_stamp)
            # print("数据",df_raw)
            # print("数据values",df_raw.values)
            self.data.append(df_raw.values)
            # sample_count.append((len(df_raw) - args.seq_len + 1) // 1)
            sample_count.append((len(df_raw) - args.seq_len + 1) // (args.seq_len // 3))
            # sample_count.append(1)

        # print("可生成的样本数量：",sample_count)

        self.cum_sum = list(accumulate(sample_count)) # 计算样本数量的累积和，并存储在cum_sum列表中。
        # print("样本累计列表：",self.cum_sum)
        # print("标准化前：",self.data)
        # print("self.data的长度:", len(self.data))
        # for i, array in enumerate(self.data):
        #     print(f"第 {i} 个元素的大小:", array.shape


        # 根据参数mean和std是否为None来选择是否进行数据标准化处理，并存储一个StandardScaler对象用于后续的标准化。
        if mean is None and std is None:
            self.scaler = StandardScaler()
            self.scaler.fit(np.vstack(self.data))
            # print("经计算的均值&标准差：",self.scaler.mean,self.scaler.std)
            for x in range(len(self.data)):
                self.data[x] = self.scaler.transform(self.data[x])
        else:
            # print("均值&标准差：",mean,std)
            self.scaler = StandardScaler(mean=mean, std=std)
            for x in range(len(self.data)):
                self.data[x] = self.scaler.transform(self.data[x])

        # print("标准化后：",self.data[0])
        # print("self.data的长度:", len(self.data))
        # for i, array in enumerate(self.data):
        #     print(f"第 {i} 个元素的大小:", array.shape)


    def _parse_index(self, idx):
        '''
        用于解析索引，将全局索引转换为文件索引和偏移量。它接受一个索引idx作为输入，然后根据累积样本数量cum_sum找到对应的文件索引fidx和偏移量offset，并返回它们。
        '''
        fidx = next(i for i, cum_sum in enumerate(self.cum_sum, start=0) if cum_sum >= idx + 1)
        offset = idx - (self.cum_sum[fidx - 1] if fidx > 0 else 0)
        return fidx, offset

    def __len__(self):
        '''
        返回数据集的长度。如果as_day为True，则返回最后一个累积样本数量，否则返回数据列表的长度。
        '''
        # print(self.cum_sum)
        # print(len(self.data))
        data_len = self.cum_sum[-1] if self.as_day else len(self.data)
        # print("数据集长度：",data_len)
        return data_len

    def __getitem__(self, index):
        '''
        用于获取数据集中的单个样本。如果as_day为True，则根据索引解析文件索引和偏移量，然后返回相应的数据、时间戳和标签。
        如果as_day为False，则直接返回数据、时间戳和标签。
        '''
        if self.as_day:
            fidx, offset = self._parse_index(index)
            offset = offset * (args.seq_len // 3)
            # offset = offset * 1
            x_end = offset + args.seq_len
            return self.data[fidx][offset:x_end], self.time_stamp[fidx][offset:x_end], self.data[fidx][offset:x_end]
        else:
            label = torch.zeros((33,))
            label[self.labels[index]] = 1
            return self.data[index], self.time_stamp[index], label

# expknn中用了FinalData
class FinalData(Dataset):
    def __init__(self, start, end):
        final_data = pd.DataFrame(
            pd.read_csv(os.path.join(args.dataset_dir, 'final_data.csv'), encoding='ISO-8859-1')).replace('â\\x80?',
                                                                                                          '-')
        self.data = final_data.drop(labels='recommendation', axis=1).to_numpy()[start:end]
        self.labels = final_data.loc[:, 'recommendation'].to_numpy()[start:end]

    def __len__(self):
        # 返回数据集的长度
        return len(self.data)

    def __getitem__(self, index):
        # 用于获取单个样本数据
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
