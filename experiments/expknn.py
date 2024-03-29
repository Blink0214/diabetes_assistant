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

literals = [
    'increase_basal',
    'decrease_basal',
    'no_adjustment_recommended',
    'increase AM bolus',
    'decrease AM bolus',
    'increase PM bolus',
    'decrease PM bolus',
    'increase evening bolus',
    'decrease evening bolus',
    'increase overnight bolus',
    'decrease overnight bolus',
]


def get_dummies(labels):
    nums = []
    for s in labels:
        for idx, l in enumerate(literals, start=0):
            if s == l:
                nums.append(idx)
    return nums


class Expknn(Exp):
    def __init__(self, name="knn", **kwargs):
        for key, value in kwargs.items():
            setattr(args, key, value)

        setting = f"{name}-seed{args.seed}"
        super(Expknn, self).__init__(setting)
        self.params['args'] = args

    def _get_data(self):
        train_dataset = FinalData(0, 1080)
        vali_dataset = FinalData(1080, 1215)
        test_dataset = FinalData(1215, 1350)

        label_counts = [1.0] * len(literals)
        self.weights = [0] * len(literals)
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
        pred = self.model.predict(self.vali_dataset)
        corrects = torch.sum(pred == self.vali_label).item()
        print('acc', corrects / len(pred))

    def _build_model(self):
        model = KNN(k=26, classes=11)
        model.fit(self.train_dataset, self.train_label, self.weights)
        return model
