import os
import logging as log

import pandas as pd
import torch
from torch import optim, nn
from torch.utils.data import DataLoader
from config.args import args, device
from data.dataset import SIMU4AE
from experiments.exp import Exp
from experiments.expknn import get_dummies, literals
from model.classifier import CLS
from model.knn import KNN
from model.you_know_who import YKW


class Expsimu(Exp):
    def __init__(self, name="simu", **kwargs):
        for key, value in kwargs.items():
            setattr(args, key, value)

        setting = f"{name}_em{args.num_embed}-hd{args.num_hidden}-seq{args.seq_len}"
        super(Expsimu, self).__init__(setting)
        self.params['args'] = args

        self.loss_func = nn.MSELoss

    def _get_data(self):
        directory_path = str(os.path.join(args.dataset_dir, 'data'))
        label_path = str(os.path.join(args.dataset_dir, 'extracted_values.csv'))
        df = pd.read_csv(label_path)
        train_file = []
        train_label = []
        val_file = []
        val_label = []
        test_file = []
        test_label = []

        one_hot_encoded = get_dummies(df['ExtractedValue'])
        args.classes = len(literals)

        for i, (index, row) in enumerate(df.iterrows(), start=1):
            path = os.path.join(directory_path, row['FileName'] + '.csv')
            label = one_hot_encoded[index]
            if i % 10 == 0:
                test_file.append(path)
                test_label.append(label)
            elif i % 10 == 9:
                val_file.append(path)
                val_label.append(label)
            else:
                train_file.append(path)
                train_label.append(label)

        train_dataset = SIMU4AE(train_file, train_label)
        mean, std = train_dataset.scaler.mean, train_dataset.scaler.std
        self.params['mean'], self.params['std'] = mean, std
        vali_dataset = SIMU4AE(val_file, val_label, mean, std)
        test_dataset = SIMU4AE(test_file, test_label, mean, std)

        self.train_dataset = train_dataset
        self.vali_dataset = vali_dataset
        self.train_label = torch.tensor(train_label, dtype=torch.int).to(device)
        self.val_label = torch.tensor(val_label, dtype=torch.int).to(device)

        self.train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
        self.vali_loader = DataLoader(vali_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
        self.test_loader = DataLoader(test_dataset, batch_size=args.batch_size, drop_last=True)

    def next_phase(self):
        encoder = self.model
        self.model = CLS(scale_size=args.isometric_kernel, encoder=encoder).float().to(device)
        self.train_dataset.as_day = False
        self.vali_dataset.as_day = False
        self.loss_func = nn.CrossEntropyLoss

        self.setting = f"ykwcls_em{args.num_embed}-hd{args.num_hidden}-seq{args.seq_len}-cls{args.classes}"
        _path = os.path.join(args.checkpoints, self.setting)
        if not os.path.exists(_path):
            os.makedirs(_path)

        print(self.model)
        total_trainable_params = sum(
            p.numel() for p in self.model.parameters() if p.requires_grad)
        log.info(f'training parameters {total_trainable_params} ')

        args.train_epochs = 30
        args.adjust_learning_rate = False
        args.early_stop = False
        args.learning_rate = 0.01
        self.train()
        self.model.eval()
        predicts = []
        labels = []
        for data, time_stamp, label in zip(self.vali_dataset.data, self.vali_dataset.time_stamp, self.val_label):
            l = len(data) - (len(data) % args.seq_len)
            x = torch.tensor(data[:l]).unsqueeze(0).reshape((-1, data.shape[1])).float().to(device)
            mark = torch.tensor(time_stamp[:l]).unsqueeze(0).reshape(
                (-1, time_stamp.shape[1])).float().to(device)
            pred = self.model(x.unsqueeze(0), mark.unsqueeze(0))
            predicted = torch.argmax(pred, dim=1).item()
            predicts.append(predicted)
            labels.append(label.item())

        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        print('accuracy_score', accuracy_score(labels, predicts))
        print('precision_score', precision_score(labels, predicts, average='weighted'))
        print('recall_score', recall_score(labels, predicts, average='weighted'))
        print('f1_score', f1_score(labels, predicts, average='weighted'))

    def test_knn(self):
        self.load_model(os.path.join(self.checkpoint_path(), 'checkpoint.pth'))
        self.model.eval()

        self.knn = KNN(k=26, classes=11)

        fusion = []
        with torch.no_grad():
            for data, time_stamp in zip(self.train_dataset.data, self.train_dataset.time_stamp):
                l = len(data) - (len(data) % args.seq_len)
                x = torch.tensor(data[:l]).unsqueeze(0).reshape((-1, args.seq_len, data.shape[1])).float().to(device)
                mark = torch.tensor(time_stamp[:l]).unsqueeze(0).reshape(
                    (-1, args.seq_len, time_stamp.shape[1])).float().to(device)
                a = self.model.lgf.encoder(self.model.embedding(x, mark))
                b = [i.mean(dim=2) for i in a]
                fusion.append(torch.stack(b, dim=2).flatten())

            self.knn.fit(torch.stack(fusion), self.train_label)

            vali = []
            for data, time_stamp in zip(self.vali_dataset.data, self.vali_dataset.time_stamp):
                l = len(data) - (len(data) % args.seq_len)
                x = torch.tensor(data[:l]).unsqueeze(0).reshape((-1, args.seq_len, data.shape[1])).float().to(device)
                mark = torch.tensor(time_stamp[:l]).unsqueeze(0).reshape(
                    (-1, args.seq_len, time_stamp.shape[1])).float().to(device)
                a = self.model.lgf.encoder(self.model.embedding(x, mark))
                b = [i.mean(dim=2) for i in a]
                vali.append(torch.stack(b, dim=2).flatten())

            pred = self.knn.predict(torch.stack(vali))
            corrects = torch.sum(pred == self.val_label).item()
            print('acc', corrects / len(pred))
            print('label', self.val_label)
            print('pred', pred)

    def _build_model(self):
        model = YKW(in_features=args.in_features, seq_len=args.seq_len,
                    num_embed=args.num_embed,
                    dropout=args.dropout, freq=args.freq, device=device,
                    conv_kernel=args.conv_kernel, isometric_kernel=args.isometric_kernel).float().to(device)
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
        criterion = self.loss_func()
        return criterion(pred, true)

    def _train_loader(self):
        return self.train_loader

    def _vali_loader(self):
        return self.vali_loader

    def _test_loader(self):
        return self.test_loader
