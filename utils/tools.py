import os

import numpy as np
import torch
from config.args import args
import logging as log


def adjust_learning_rate(optimizer, epoch, adj_type='type1'):
    # lr = args.learning_rate * (0.2 ** (epoch // 2))
    if adj_type == 'type1':
        lr_adjust = {epoch: args.learning_rate * (0.75 ** ((epoch - 1) // 1))}
    elif adj_type == 'type2':
        lr_adjust = {
            2: 5e-4, 4: 1e-4, 6: 5e-5, 8: 1e-5,
            10: 5e-6, 15: 1e-6, 20: 5e-7
        }
    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        log.info('Updating learning rate to {}'.format(lr))


class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model, path, params=None):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path, params)
        elif score < self.best_score + self.delta:
            self.counter += 1
            log.info(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path, params)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, checkpoint_path, params):
        if self.verbose:
            log.info(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save({
            'model_state_dict': model.state_dict(),
            'params': params
        }, os.path.join(checkpoint_path, 'checkpoint.pth'))
        self.val_loss_min = val_loss


class StandardScaler:
    def __init__(self, mean=0., std=1.):
        self.mean = mean
        self.std = std

    def fit(self, data):
        self.mean = data.mean(0)
        self.std = data.std(0)

    def transform(self, data):
        mean = torch.from_numpy(self.mean).type_as(data).to(data.device) if torch.is_tensor(data) else self.mean
        std = torch.from_numpy(self.std).type_as(data).to(data.device) if torch.is_tensor(data) else self.std
        return (data - mean) / std

    def inverse_transform(self, data):
        mean = torch.from_numpy(self.mean).type_as(data).to(data.device) if torch.is_tensor(data) else self.mean
        std = torch.from_numpy(self.std).type_as(data).to(data.device) if torch.is_tensor(data) else self.std
        return (data * std) + mean
