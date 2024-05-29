import math
import os
import logging as log

import pandas as pd
import torch
from torch import optim, nn
from torch.utils.data import DataLoader
from config.args import args, device
from data.dataset import FinalData
from experiments.exp import Exp
from model.knn import KNN

# 调整方式_模拟器
# literals = [
#     'increase_basal',
#     'decrease_basal',
#     'no_adjustment_recommended',
#     'increase AM bolus',
#     'decrease AM bolus',
#     'increase PM bolus',
#     'decrease PM bolus',
#     'increase evening bolus',
#     'decrease evening bolus',
#     'increase overnight bolus',
#     'decrease overnight bolus',
# ]

''' 具体方案_删减_一天
literals = [
    'YW 5',
    'YW 2、5',
    'YW 4',
    'YW 1、5',
    'YW 1',
    'YW 1、2、4、5',
    'YW 1、3、5',
    'YW 2',
    'CX 5、7 - 10',
    'YH 0 - 14 - 7',
    'YW 3、5、6',
    'CX 5 - 8',
    'YH 1 - 12 - 9',
    'YW 4、5',
    'CX 5 - 4',
    'CX 1 - 6',
    'YW 1、2、5',
    'RN 0 - 10 - 6 - 8 - 16',
    'RN 0 - 5 - 5 - 5 - 6',
    'YH 0 - 10 - 6',
    'YH 5 - 8',
    'CX 1、2、5 - 4 - 6',
    'YW 1、2、3',
    'RN 1 - 6 - 6 - 6 - 6',
    'CX 1、2、4 - 18',
    'RN 1 - 5 - 5 - 7 - 10',
    'CX 1 - 8',
    'YW 1、2、4',
    'YH 5 - 15 - 14',
    'YH 0 - 28 - 14',
    'YW 1、2',
    'YW 1、4',
    'RN 1 - 12 - 3 - 5 - 14',
    'RN 1 - 8 - 4 - 6 - 16',
    'YW 5、8',
]
'''

'''具体方案_完整
literals = [
    'YW 5',
    'YW 2、5',
    'YW 4',
    'YW 1、5',
    '0',
    'YW 1',
    'CX 5、7 - 10',
    'YW 1、2、4',
    'YW 1、2、4、5',
    'YW 1、3、5',
    'YW 2',
    'RN 0 - 10 - 6 - 6 - 14',
    'RN 1 - 4 - 4 - 2 - 4',
    'RN 1 - 10 - 5 - 5 - 10',
    'RN 0 - 5 - 5 - 5 - 6',
    'YH 0 - 10 - 6',
    'YH 5 - 8 - 4',
    'CX 5 - 6',
    'CX 5 - 4',
    'CX 1 - 6',
    'CX 1 - 8',
    'CX 1、2、5 - 4 - 6',
    'YW 1、2、3',
    'YH 0 - 14 - 8',
    'CX 5 - 8',
    'RN 1 - 6 - 6 - 6 - 6',
    'RN 1 - 12 - 3 - 5 - 14',
    'YH 1 - 12 - 9',
    'RN 0 - 6 - 3 - 4 - 6',
    'RN 1 - 5 - 5 - 7 - 14',
    'RN 0 - 10 - 6 - 8 - 16',
    'YH 1、5 - 10 - 6',
    'YH 0 - 16 - 14',
    'YH 5 - 16 - 14',
    'YH 0 - 16 - 8 - 16',
    'YH 0 - 28 - 14',
    'YW 4、5',
    'YW 3、5、6',
    'YW 1、2、5',
    'YW 1、2',
    'YW 3、6',
    'CX 3、5 - 8',
    'YW 1、3、5、6',
    'RN 1 - 10 - 6 - 5 - 12',
    'YH 5 - 8',
    'RN 0 - 8 - 5 - 5 - 7',
    'YH 0 - 16 - 6',
    'RN 1 - 8 - 4 - 6 - 17',
    'YH 0 - 16 - 8 - 12',
    'YW 5、8',
    'RN 0 - 5 - 5 - 5 - 15',
    'RN 1、5 - 7 - 4 - 4 - 11',
    'YH 0 - 17 - 9',
    'YW 9',
    'RN 0 - 9 - 3 - 4 - 19',
    'RN 0 - 10 - 3 - 4 - 19',
    'RN 0 - 9 - 4 - 4 - 13',
    'RN 0 - 8 - 4 - 4 - 6',
    'CX 1、5 - 14',
    'YH 1、5 - 16 -12',
    'YW 2、3、5',
    'RN 0 - 8 - 4 - 6 - 12',
    'YW 1、5、6',
    'RN 1 - 8 - 6 - 7 - 8',
    'RN 1、5、7 - 8 - 6 - 8 - 10',
    'RN 1 - 14 - 4 - 4 - 13',
    'RN 0 - 11 - 8 - 9 - 19',
    'RN 1 - 17 - 7 - 4 - 11',
    'RN 1 - 6 - 9 - 7 - 15',
    'RN 1 - 13 - 11 - 13 - 16',
    'RN 1、5 - 9 - 6 - 4 - 16',
    'YH 5、6 - 22 - 10',
    'RN 1 - 11 - 7 - 9 - 18',
    'RN 1 - 7 - 7 - 4 - 16',
    'RN 1、5 - 11 - 7 - 7 - 19',
    'RN 1、3、4 - 9 - 6 - 4 - 13',
    'RN 0 - 12 - 12 - 6 - 6',
    'RN 0 - 15 - 7 - 8 - 14',
    'RN 1、5 - 1 - 8 - 8 - 17',
    'RN 3 - 7 - 4 - 8 - 7',
    'RN 1、5 - 7 - 6 - 6 - 6',
    'CX 5 - 12',
    'RN 1 - 7 - 4 - 7 - 16',
    'RN 0 - 9 - 6 - 7 - 21',
    'RN 0 - 8 - 4 - 0 - 13',
    'RN 1 - 7 - 4 - 6 - 14',
    'RN 1 - 8 - 4 - 4 - 15',
    'RN 0 - 10 - 9 - 8 - 13',
    'CX 1、5 - 8',
    'RN 0 - 8 - 6 - 8 - 21',
    'RN 1、5 - 9 - 4 - 4 - 12',
    'YH 5 - 8 - 4 - 4 - 15',
    'RN 0 - 11 - 4 - 3 - 20',
]'''

# 胰岛素_删减_三天  35
'''
literals = [
    '0',
    'CX 5、7 - 10',
    'RN 0 - 9 - 3 - 4 - 19',
    'RN 0 - 10 - 6 - 6 - 14',
    'RN 1 - 4 - 4 - 2 - 4',
    'RN 1 - 10 - 5 - 5 - 10',
    'RN 0 - 5 - 5 - 5 - 6',
    'YH 0 - 10 - 6',
    'YH 5 - 8 - 4',
    'CX 5 - 6',
    'CX 5 - 4',
    'CX 1 - 6',
    'CX 1 - 8',
    'CX 1、2、5 - 4 - 6',
    'YH 0 - 14 - 8',
    'CX 5 - 8',
    'RN 1 - 6 - 6 - 6 - 6',
    'RN 1 - 12 - 3 - 5 - 14',
    'YH 1 - 12 - 9',
    'RN 0 - 6 - 3 - 4 - 6',
    'RN 1 - 5 - 5 - 7 - 14',
    'RN 0 - 10 - 6 - 8 - 16',
    'YH 1、5 - 10 - 6',
    'YH 0 - 16 - 14',
    'YH 5 - 16 - 14',
    'YH 0 - 16 - 8 - 16',
    'YH 0 - 28 - 14',
    'CX 3、5 - 8',
    'RN 1 - 10 - 6 - 5 - 12',
    'YH 5 - 8',
    'RN 0 - 8 - 5 - 5 - 7',
    'YH 0 - 16 - 6',
    'RN 1 - 8 - 4 - 6 - 17',
    'YH 0 - 16 - 8 - 12',
    'RN 0 - 5 - 5 - 5 - 15',
]
'''
# 纯药物_三天  20
'''
literals = [
    'YW 5',
    'YW 2、5',
    'YW 4',
    'YW 1、5',
    'YW 1',
    'YW 1、2、4',
    'YW 1、2、4、5',
    'YW 1、3、5',
    'YW 2',
    'YW 1、2、3',
    'YW 4、5',
    'YW 3、5、6',
    'YW 1、2、5',
    'YW 1、2',
    'YW 3、6',
    'YW 1、3、5、6',
    'YW 5、8',
    'YW 9',
    'YW 2、3、5',
    'YW 1、5、6',
]'''

# 具体方案_删减_三天
literals = [
    'YW 5',
    'YW 2、5',
    'YW 4',
    'YW 1、5',
    '0',
    'YW 1',
    'CX 5、7 - 10',
    'YW 1、2、4',
    'YW 1、2、4、5',
    'YW 1、3、5',
    'RN 0 - 9 - 3 - 4 - 19',
    'YW 2',
    'RN 0 - 10 - 6 - 6 - 14',
    'RN 1 - 4 - 4 - 2 - 4',
    'RN 1 - 10 - 5 - 5 - 10',
    'RN 0 - 5 - 5 - 5 - 6',
    'YH 0 - 10 - 6',
    'YH 5 - 8 - 4',
    'CX 5 - 6',
    'CX 5 - 4',
    'CX 1 - 6',
    'CX 1 - 8',
    'CX 1、2、5 - 4 - 6',
    'YW 1、2、3',
    'YH 0 - 14 - 8',
    'CX 5 - 8',
    'RN 1 - 6 - 6 - 6 - 6',
    'RN 1 - 12 - 3 - 5 - 14',
    'YH 1 - 12 - 9',
    'RN 0 - 6 - 3 - 4 - 6',
    'RN 1 - 5 - 5 - 7 - 14',
    'RN 0 - 10 - 6 - 8 - 16',
    'CX 3、5 - 8',
]


# 调整方式_三天
'''literals = [
    'no adjustment',
    'decrease basal dose',
    'increase breakfast bolus',
    'increase basal dose',
    'no cure',
    'increase dinner bolus',
    'decrease dinner bolus',
    'decrease breakfast bolus',
    'decrease lunch bolus',
    'increase lunch bolus',
    'decrease all dose',
    'increase all dose',
]'''


def get_dummies(labels):
    '''
    接受一个标签列表作为输入，将标签列表中的字符串标签转换为对应的数字索引列表。
    例如，如果输入是['increase_basal', 'no_adjustment_recommended']，则输出可能是[0, 2]，
    其中0对应'increase_basal'，2对应'no_adjustment_recommended'。
    这种转换通常用于将分类标签映射到模型能够处理的数字形式。
    '''
    nums = []
    if_match = False
    for index, s in enumerate(labels):
        if_match = False
        for idx, l in enumerate(literals, start=0):
            if s == l:
                if_match = True
                nums.append(idx)
        if if_match == False:
            print("未匹配上：",index,s)
    print("独热编码个数:",len(nums))
    return nums


class Expknn(Exp):
    '''
    承自Exp类的子类。它重写了__init__、_get_data、test和_build_model方法。
    该类用于管理k近邻算法的实验设置和执行。
    '''
    def __init__(self, name="knn", **kwargs):
        for key, value in kwargs.items():
            setattr(args, key, value)

        setting = f"{name}-seed{args.seed}"
        super(Expknn, self).__init__(setting)
        self.params['args'] = args

    def _get_data(self): 
        '''
        用于获取数据集。它创建了训练、验证和测试数据集的实例，并将其加载到相应的数据张量中。
        此外，计算了类别权重weights，并将其用于后续模型训练过程中的类别不平衡处理。
        '''
        train_dataset = FinalData(0, 1080)
        vali_dataset = FinalData(1080, 1215)
        test_dataset = FinalData(1215, 1350)

        label_counts = [1.0] * len(literals)
        self.weights = [1] * len(literals)
        all_count = 1.0
        for recommendation in train_dataset.labels:
            for idx, l in enumerate(literals, start=0):
                if recommendation == l:
                    label_counts[idx] += 1
            all_count += 1
        count = 1
        for i in range(len(self.weights)):
            self.weights[i] = count / label_counts[i]

        self.train_dataset = torch.Tensor(train_dataset.data).to(device)
        self.train_label = torch.tensor(get_dummies(train_dataset.labels)).to(device)
        self.vali_dataset = torch.Tensor(vali_dataset.data).to(device)
        self.vali_label = torch.tensor(get_dummies(vali_dataset.labels)).to(device)

    def test(self):
        '''
        测试模型在验证集上的性能。它使用训练好的模型对验证数据集进行预测，并计算分类准确率。
        '''
        pred = self.model.predict(self.vali_dataset)
        corrects = torch.sum(pred == self.vali_label).item()
        print('acc', corrects / len(pred))

    def _build_model(self):
        '''
        用于构建k近邻模型。在这个方法中，创建了一个KNN类的实例，指定了k值和类别数量，并调用其fit方法来训练模型。最后返回训练好的模型。
        '''
        # model = KNN(k=26, classes=11)
        # model = KNN(k=26, classes=35)
        # model = KNN(k=26, classes=12)
        # model = KNN(k=26, classes=93)
        model = KNN(k=26, classes=42)
        model.fit(self.train_dataset, self.train_label, self.weights)
        return model
