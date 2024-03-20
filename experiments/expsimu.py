import os
import logging as log

import pandas as pd
from torch import optim, nn
from torch.utils.data import DataLoader
from config.args import args, device
from data.dataset import SIMU
from experiments.exp import Exp
from model.rnn import SimpleRNN
from model.you_know_who import YKW


class Expsimu(Exp):
    def __init__(self, name="simu", **kwargs):
        for key, value in kwargs.items():
            setattr(args, key, value)

        setting = f"{name}_if{args.in_features}_cls{args.classes}-seed{args.seed}"
        super(Expsimu, self).__init__(setting, args.model)
        self.params['args'] = args

    def _get_data(self):
        directory_path = str(os.path.join(args.dataset_dir, args.subject))
        label_path = str(os.path.join(args.dataset_dir, args.subject + '.csv'))
        df = pd.read_csv(label_path)
        train_file = []
        train_label = []
        val_file = []
        val_label = []
        test_file = []
        test_label = []

        one_hot_encoded = pd.get_dummies(df['ExtractedValue'])
        args.classes = one_hot_encoded.shape[1]

        for i, (index, row) in enumerate(df.iterrows(), start=1):
            path = os.path.join(directory_path, row['FileName'] + '.csv')
            label = one_hot_encoded.loc[index].values
            if i % 10 == 0:
                test_file.append(path)
                test_label.append(label)
            elif i % 10 == 9:
                val_file.append(path)
                val_label.append(label)
            else:
                train_file.append(path)
                train_label.append(label)

        train_dataset = SIMU(train_file, train_label)
        mean, std = train_dataset.scaler.mean, train_dataset.scaler.std
        self.params['mean'], self.params['std'] = mean, std
        vali_dataset = SIMU(val_file, val_label, mean, std)
        test_dataset = SIMU(test_file, test_label, mean, std)

        self.train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
        self.vali_loader = DataLoader(vali_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
        self.test_loader = DataLoader(test_dataset, batch_size=args.batch_size, drop_last=True)

    def _build_model(self, model_str):
        if model_str == 'ykw':
            model = YKW(in_features=args.in_features, seq_len=args.seq_len, classes=args.classes,
                        num_hidden=args.num_hidden, lgf_layers=args.lgf_layers,
                        dropout=args.dropout, freq=args.freq, device=device,
                        decomp_kernel=args.decomp_kernel, conv_kernel=args.conv_kernel,
                        isometric_kernel=args.isometric_kernel).float().to(device)
        elif model_str == 'rnn':
            model = SimpleRNN(args.in_features, args.num_hidden, args.classes).float().to(device)
        total_params = sum(p.numel() for p in model.parameters())
        log.info(f'total parameters {total_params} ')
        total_trainable_params = sum(
            p.numel() for p in model.parameters() if p.requires_grad)
        log.info(f'training parameters {total_trainable_params} ')

        if args.use_multi_gpu:
            model = nn.DataParallel(model, device_ids=args.devices)
        return model

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=args.learning_rate)
        return model_optim

    def _loss_function(self, pred, true):
        criterion = nn.CrossEntropyLoss()
        return criterion(pred, true)

    def _train_loader(self):
        return self.train_loader

    def _vali_loader(self):
        return self.vali_loader

    def _test_loader(self):
        return self.test_loader
