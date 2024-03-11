import os
import logging as log
from torch import optim, nn
from torch.utils.data import DataLoader
from config.args import args, device
from experiments.exp import Exp
from model.micn import MICN
from data.dataset import SHT2


class ExpSHT2(Exp):
    def __init__(self, name="sht2", **kwargs):
        for key, value in kwargs.items():
            setattr(args, key, value)

        setting = f"{name}_wth_sl{args.seq_len}_pl{args.pred_len}_if{args.in_features}_of{args.out_features}_freq_{args.freq}-seed{args.seed}"
        super(ExpSHT2, self).__init__(setting)
        self.params['args'] = args

    def _get_data(self):
        directory_path = str(os.path.join(args.dataset_dir, args.dataset_path))
        flist = [os.path.join(directory_path, f) for f in os.listdir(directory_path)]
        num_train = int(len(flist) * 0.8)
        num_test = int(len(flist) * 0.1)
        num_vali = len(flist) - num_train - num_test
        ftrain = flist[:num_train]
        fval = flist[num_train:num_train + num_vali]
        ftest = flist[num_train + num_vali:]

        train_dataset = SHT2(ftrain)
        mean, std = train_dataset.scaler.mean, train_dataset.scaler.std
        self.params['mean'], self.params['std'] = mean, std
        vali_dataset = SHT2(fval, mean, std)
        test_dataset = SHT2(ftest, mean, std)

        self.train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
        self.vali_loader = DataLoader(vali_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
        self.test_loader = DataLoader(test_dataset, batch_size=args.batch_size, drop_last=True)

    def _build_model(self):
        model = MICN(in_features=args.in_features, out_features=args.out_features,
                     seq_len=args.seq_len, pred_len=args.pred_len, num_hidden=args.num_hidden,
                     mic_layers=args.mic_layers, dropout=args.dropout, freq=args.freq, device=device,
                     decomp_kernel=args.decomp_kernel, conv_kernel=args.conv_kernel,
                     isometric_kernel=args.isometric_kernel).float().to(device)
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
        criterion = nn.MSELoss()
        return criterion(pred, true)

    def _train_loader(self):
        return self.train_loader

    def _vali_loader(self):
        return self.vali_loader

    def _test_loader(self):
        return self.test_loader
