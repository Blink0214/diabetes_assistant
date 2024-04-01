import logging as log
import os
import shutil
import time

import numpy as np
import torch
from torch import nn

from config.args import args, device
from utils.tools import EarlyStopping, adjust_learning_rate


class Exp(object):
    def __init__(self, setting: str):
        log.info(args)
        self.setting = setting
        self.params = {}
        _path = os.path.join(args.checkpoints, setting)
        if not os.path.exists(_path):
            os.makedirs(_path)
        self._get_data()
        self.model = self._build_model()
        print(self.model)

    def _train_loader(self):
        raise NotImplementedError

    def _vali_loader(self):
        raise NotImplementedError

    def _test_loader(self):
        raise NotImplementedError

    def _init_knn(self):
        raise NotImplementedError

    def _build_model(self):
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
        acc = 0

        correct = 0
        total = 0
        for i, (data, mark, label) in enumerate(self._vali_loader()):
            x = data.float().to(device)
            label = label.float().to(device)
            total += len(label)

            pred = self.model(x, mark.float().to(device))
            predicted = torch.argmax(pred, dim=1)
            labels = torch.argmax(label, dim=1)
            correct += (predicted == labels).sum().item()

            loss = self._loss_function(pred, label)
            total_loss.append(loss.item())

        if self.loss_func is nn.CrossEntropyLoss:
            acc = (100 * correct / total)

        vali_loss = np.average(total_loss)
        self.model.train()

        return vali_loss, acc

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

            iter_count = 0
            train_loss = []
            self.model.train()
            epoch_time = time.time()
            for i, (data, mark, label) in enumerate(self._train_loader()):
                iter_count += 1
                x = data.float().to(device)

                model_optim.zero_grad()
                pred = self.model(x, mark.float().to(device))
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

            vali_loss, acc = self.vali()

            log.info("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss) + (f" Acc: {acc:.2f}" if acc != 0 else ""))
            early_stopping(vali_loss, self.model, self.checkpoint_path(), self.params)
            if early_stopping.early_stop and args.early_stop:
                log.info("Early stopping")
                break

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
