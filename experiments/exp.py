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
    '''
    定义了一个实验类Exp，用于模型的训练、验证和测试。
    '''
    def __init__(self, setting: str):
        log.info(args)
        self.setting = setting
        self.params = {}
        _path = os.path.join(args.checkpoints, setting)
        if not os.path.exists(_path):
            os.makedirs(_path)
        self._get_data() # 准备数据
        self.model = self._build_model() # 构建模型
        print(self.model)

    # 均为抽象方法，需要在子类中实现。
    # 定义训练、验证和测试数据集的加载器
    def _train_loader(self):
        raise NotImplementedError

    def _vali_loader(self):
        raise NotImplementedError

    def _test_loader(self):
        raise NotImplementedError

    # 初始化KNN模型
    def _init_knn(self):
        raise NotImplementedError
    # 构建模型
    def _build_model(self):
        raise NotImplementedError
    # 获取数据
    def _get_data(self):
        raise NotImplementedError
    # 选择优化器
    def _select_optimizer(self):
        raise NotImplementedError
    # 定义损失函数
    def _loss_function(self, pred, true):
        raise NotImplementedError
    # 返回实验的检查点路径
    '''
    在训练过程中，通常会周期性地保存模型的参数，以便在需要时可以从中恢复模型的状态，或者用于后续的评估和测试。这些保存的模型状态通常被称为检查点。
    因此，checkpoint_path(self)方法的作用就是生成当前实验的检查点路径。这个路径通常由实验的设置信息和固定的路径部分组成，以便可以唯一地标识一个实验的检查点文件。例如，检查点路径可能会包含实验的名称、日期时间信息等。
    通过调用这个方法，可以方便地获取当前实验的检查点路径，以便进行模型参数的保存或加载。
    '''
    def checkpoint_path(self):
        return os.path.join(args.checkpoints, self.setting)
    # 加载模型的方法，接受一个模型路径作为参数，加载模型的参数和状态字典。
    def load_model(self, model_path):
        checkpoint = torch.load(model_path)
        self.params = checkpoint['params']
        self.model.load_state_dict(checkpoint['model_state_dict'])

    def vali(self):
        '''
        在验证集上进行模型评估的方法。返回验证集上的损失和准确率。
        '''
        self.model.eval() # 将模型切换到评估模式，这会影响模型的行为，例如在BatchNormalization和Dropout等层中，会禁用随机失活和批量归一化。
        total_loss = [] # 初始化用于存储总损失和准确率的变量。
        acc = 0

        correct = 0 # 初始化正确预测的样本数量和总样本数量的变量。
        total = 0
        # 通过验证集加载器迭代加载验证数据。
        for i, (data, mark, label) in enumerate(self._vali_loader()): 
            x = data.float().to(device)
            label = label.float().to(device)
            total += len(label)

            pred = self.model(x, mark.float().to(device))
            predicted = torch.argmax(pred, dim=1) # 根据预测的概率分布，选择概率最高的类别作为预测结果。
            labels = torch.argmax(label, dim=1)
            '''
            用于在张量（tensor）中沿着指定的维度找到最大值的索引。具体来说，torch.argmax(input, dim=None, keepdim=False) 的作用是：
            input: 输入的张量，可以是任意形状的张量。
            dim: 沿着哪个维度进行查找最大值的索引，默认为 None，表示在整个张量中查找最大值的索引。
            keepdim: 如果设置为 True，则结果张量会保持与输入张量相同的维度，即在最大值所在维度上保持维度为 1。默认为 False。
            '''

            correct += (predicted == labels).sum().item() # 计算正确预测的样本数量。

            loss = self._loss_function(pred, label) # 计算损失并将其添加到总损失列表中。
            total_loss.append(loss.item())

        if self.loss_func is nn.CrossEntropyLoss:
            acc = (100 * correct / total) # 如果损失函数为交叉熵损失，则计算准确率。

        vali_loss = np.average(total_loss) # 计算平均验证损失。
        self.model.train() # 将模型切换回训练模式。

        return vali_loss, acc # 返回验证损失和准确率。

    def train(self):
        '''
        模型训练方法。在训练过程中，使用早停策略EarlyStopping，并保存最佳模型的参数。同时，根据需要调整学习率。
        '''
        time_now = time.time() # 记录当前时间，用于计算每个训练批次的时间。

        train_steps = len(self._train_loader()) # 获取训练集的总批次数，以便在训练过程中进行迭代。
        early_stopping = EarlyStopping(patience=args.patience, verbose=True) # 创建早停对象，用于在训练过程中监视验证损失，并在损失停止减小时停止训练。

        model_optim = self._select_optimizer() # 选择优化器，根据参数配置选择合适的优化器来更新模型的参数。

        if args.use_amp:
            scaler = torch.cuda.amp.GradScaler() # 如果启用了混合精度训练（Automatic Mixed Precision），则创建一个梯度缩放器，用于动态调整梯度的缩放比例。

        best_model_path = os.path.join(self.checkpoint_path(), 'checkpoint.pth') # 生成最佳模型的保存路径。
        for epoch in range(args.train_epochs): # 开始迭代训练过程，遍历所有的训练周期（epochs）。
            log.debug("=========================   epoch start   =========================") # 打印调试信息，表示开始一个新的训练周期。

            # 初始化迭代计数器、存储训练损失的列表，并将模型切换到训练模式。同时记录当前周期的开始时间。
            iter_count = 0
            train_loss = []
            self.model.train()
            epoch_time = time.time()
            # 通过训练集加载器迭代加载训练数据。
            for i, (data, mark, label) in enumerate(self._train_loader()):
                # # 打印当前批次的数据
                # print(f"Batch {i + 1}:")
                # print("Data:", data.shape)
                # print("Mark:", mark.shape)
                # print("Label:", label.shape)


                iter_count += 1
                x = data.float().to(device) # 将数据转换为浮点张量，并将其移动到指定的设备上。
 
                model_optim.zero_grad() # 清零优化器的梯度，以准备计算新的梯度。
                pred = self.model(x, mark.float().to(device)) # 通过模型进行前向传播，得到预测结果。
                # print("torch形状：",pred.shape)
                loss = self._loss_function(pred, label.float().to(device)) # 计算损失。将损失值添加到训练损失列表中。
                train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    # 每经过100个批次，打印训练损失信息，并估算训练剩余时间。
                    log.info("\tbatches: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((args.train_epochs - epoch) * train_steps - i)
                    log.info('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                # 根据损失值计算梯度并更新模型参数。如果启用了混合精度训练，则通过梯度缩放器来缩放梯度。
                if args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    model_optim.step()

            log.info("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time)) # 打印每个周期的耗时信息。

            train_loss = np.average(train_loss) # 计算平均训练损失。

            vali_loss, acc = self.vali() # 在当前周期结束后，通过验证集评估模型，并获取验证损失和准确率。

            log.info("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss) + (f" Acc: {acc:.2f}" if acc != 0 else "")) # 打印训练和验证损失信息。
            early_stopping(vali_loss, self.model, self.checkpoint_path(), self.params) # 调用早停对象，判断是否需要提前停止训练。
            if early_stopping.early_stop and args.early_stop:
                log.info("Early stopping") # 如果早停条件满足，则提前停止训练。
                break

            adjust_learning_rate(model_optim, epoch + 1) # 调整学习率，根据参数设置动态调整学习率。

            shutil.copy2(best_model_path, os.path.join(self.checkpoint_path(), f'checkpoint-epoch{epoch + 1}.pth')) # 复制当前最佳模型到特定路径，命名为当前周期的检查点。
            if epoch != 0:
                os.remove(os.path.join(self.checkpoint_path(), f'checkpoint-epoch{epoch}.pth')) # 删除上一个周期保存的检查点文件。
            log.debug("=========================   epoch end   =========================") # 打印调试信息，表示当前周期训练结束。

        checkpoint = torch.load(best_model_path)
        self.params = checkpoint['params']
        self.model.load_state_dict(checkpoint['model_state_dict'])
        # 在所有周期结束后，加载最佳模型的参数。

    def test(self):
        '''
        试模型的方法。这个方法在基类中被定义为空，需要在子类中进行具体实现。
        '''
        pass
