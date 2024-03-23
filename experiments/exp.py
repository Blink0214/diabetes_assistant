import logging as log
import os
import shutil
import time

import numpy as np
import torch
from config.args import args, device
from utils.tools import EarlyStopping, adjust_learning_rate


class KNN:
    def __init__(self, k=26):
        self.k = k

    def fit(self, X, y):
        self.X_train = X.to(device)
        self.y_train = torch.argmax(y.to(device), 1)

    def predict(self, X):
        y_pred = [self._predict(x) for x in X]
        return torch.stack(y_pred, dim=0)

    def _predict(self, x):
        # 计算x与所有训练样本的欧氏距离
        distances = [torch.norm(x - x_train) for x_train in self.X_train]
        # 找到K个最近邻居的索引
        k_indices = torch.topk(torch.tensor(distances), self.k, largest=False).indices
        # 获取K个最近邻居的标签
        k_nearest_labels = [self.y_train[i] for i in k_indices]
        # 多数表决法来预测标签
        most_common = torch.bincount(torch.tensor(k_nearest_labels)).argmax()
        one_hot_tensor = torch.zeros(args.classes)
        # Fill the corresponding index with 1
        one_hot_tensor.scatter_(0, torch.LongTensor([most_common]), 1)
        return one_hot_tensor.to(device)


class Exp(object):
    def __init__(self, setting: str, model_str: str):
        log.info(args)
        self.setting = setting
        self.params = {}
        _path = os.path.join(args.checkpoints, setting)
        if not os.path.exists(_path):
            os.makedirs(_path)
        self._get_data()
        self.model = self._build_model(model_str)
        self.knn = KNN()
        print(self.model)

    def _train_loader(self):
        raise NotImplementedError

    def _vali_loader(self):
        raise NotImplementedError

    def _test_loader(self):
        raise NotImplementedError

    def _init_knn(self):
        raise NotImplementedError

    def _build_model(self, model_str):
        raise NotImplementedError

    def _get_data(self):
        raise NotImplementedError

    def _select_optimizer(self):
        raise NotImplementedError

    def _loss_function(self, pred, true):
        raise NotImplementedError

    def checkpoint_path(self):
        return os.path.join(args.checkpoints, self.setting)

    def load_model(self, model_path):
        checkpoint = torch.load(model_path)
        self.params = checkpoint['params']
        self.model.load_state_dict(checkpoint['model_state_dict'])

    def vali(self):
        self.model.eval()
        total_loss = []
        correct = 0
        total = 0

        for i, (data, mark, label) in enumerate(self._vali_loader()):
            label = label.float().to(device)
            total += label.size(0)

            pred = self.model(data.float().to(device), mark.float().to(device)) if args.model == 'ykw' else self.model(data.float().to(device))
            _, predicted = torch.max(pred, 1)
            _, labels = torch.max(label, 1)
            print(predicted)
            correct += (predicted == labels).sum().item()

            loss = self._loss_function(pred, label)
            total_loss.append(loss.item())

        vali_loss = np.average(total_loss)
        self.model.train()

        return (100 * correct / total), vali_loss

    def train(self):
        time_now = time.time()

        train_steps = len(self._train_loader())
        early_stopping = EarlyStopping(patience=args.patience, verbose=True)

        model_optim = self._select_optimizer()

        if args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        best_model_path = os.path.join(self.checkpoint_path(), 'checkpoint.pth')
        for epoch in range(args.train_epochs):
            log.debug("=========================   epoch start   =========================")
            self._init_knn()

            iter_count = 0
            train_loss = []
            self.model.train()
            epoch_time = time.time()
            for i, (data, mark, label) in enumerate(self._train_loader()):
                iter_count += 1

                model_optim.zero_grad()
                pred = self.knn.predict(self.model(data.float().to(device), mark.float().to(device))) if args.model == 'ykw' else self.knn.predict(self.model(data.float().to(device)))
                loss = self._loss_function(pred, label.float().to(device))
                train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    log.info("\tbatches: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((args.train_epochs - epoch) * train_steps - i)
                    log.info('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                if args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    model_optim.step()

            log.info("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))

            train_loss = np.average(train_loss)

            acc, vali_loss = self.vali()

            log.info("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Acc: {4:.2f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, acc))
            early_stopping(vali_loss, self.model, self.checkpoint_path(), self.params)
            # if early_stopping.early_stop:
            #     log.info("Early stopping")
            #     break

            adjust_learning_rate(model_optim, epoch + 1)

            shutil.copy2(best_model_path, os.path.join(self.checkpoint_path(), f'checkpoint-epoch{epoch + 1}.pth'))
            if epoch != 0:
                os.remove(os.path.join(self.checkpoint_path(), f'checkpoint-epoch{epoch}.pth'))
            log.debug("=========================   epoch end   =========================")

        checkpoint = torch.load(best_model_path)
        self.params = checkpoint['params']
        self.model.load_state_dict(checkpoint['model_state_dict'])

    def test(self):
        pass
