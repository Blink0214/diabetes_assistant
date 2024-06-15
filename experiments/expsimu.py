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
from utils.timefeature import time_features
from utils.tools import StandardScaler

# from sklearn.metrics import accuracy_score

# 基于pytorch的深度学习模型的训练框架
class Expsimu(Exp):
    '''
    继承Exp
    '''
    def __init__(self, name="simu", **kwargs):
        '''
        初始化函数：初始化实验对象。
        '''
        # 遍历传入的关键字参数，使用内置函数setattr将关键字参数的值设置为args对象的属性。
        # args对象是一个命名空间，可能包含许多配置参数。包括模型超参数、数据集路径等等。
        for key, value in kwargs.items():
            setattr(args, key, value)

        # 构建一个设置字符串，包含实验名称和一些关键参数的值，用于标识实验设置。
        setting = f"{name}_em{args.num_embed}-hd{args.num_hidden}-seq{args.seq_len}"
        # 调用父类的初始化函数，传入设置字符串作为参数，以便在父类中初始化实验对象
        super(Expsimu, self).__init__(setting)
        # 将配置参数'args'存储在实验对象的参数字典中，以备后续使用。
        self.params['args'] = args
        # 将损失函数 nn.MSELoss 赋值给实验对象的 loss_func 属性。这表示在训练模型时会使用均方误差损失。
        self.loss_func = nn.MSELoss

    def _get_data(self):
        '''
        从文件中加载数据集，并划分训练集、验证集和测试集
        '''
        directory_path = str(os.path.join(args.dataset_dir, 'data')) # dataset_dir=os.path.join('.', 'datasets', 'trainset')
        # label_path = str(os.path.join(args.dataset_dir, 'extracted_values.csv'))
        label_path = str(os.path.join(args.dataset_dir, 'cure_label.csv'))
        # label_path = str(os.path.join(args.dataset_dir, 'adjust_label.csv'))
        df = pd.read_csv(label_path)
        # df = pd.read_csv(label_path,encoding='ISO-8859-1')
        train_file = []
        train_label = []
        val_file = []
        val_label = []
        test_file = []
        test_label = []

        # 对ExtractedValue列进行独热编码，将每个类别转换为一个长度等于类别数量的二进制向量，
        # 其中只有一个元素是1，其余元素均是0，表示该样本属于哪个类别。这个操作通常在分类任务中用于将分类标签转换为模型可接受的格式。
        # one_hot_encoded = get_dummies(df['ExtractedValue']) 
        # one_hot_encoded = get_dummies(df['adjusted cure']) 
        one_hot_encoded = get_dummies(df['label'])

        args.classes = len(literals) # 设置模型的类别数量。
        '''
        literals 是一个列表或集合，其中包含了所有可能的类别（类别标签）。
        len(literals) 返回了类别的数量，并将其赋值给 args.classes。
        在训练过程中，这个参数通常用于指定模型的输出层的神经元数量，以适应任务中的类别数量。
        '''

        # df.iterrows返回一个迭代器，可以产生一个元组序列，每个元组包含两个值：行索引（index）和相应的行数据（row）。i是迭代的计数器，从1开始。
        for i, (index, row) in enumerate(df.iterrows(), start=1):
            # path = os.path.join(directory_path, row['FileName'] + '.csv')
            # path = os.path.join(directory_path, row['file name'])
            path = os.path.join(directory_path, row['file_name'])
            # print("标签号：",index)
            label = one_hot_encoded[index] # 获取当前样本使用独热编码后的标签
            # # 每10个样本中的前8个添加到训练集，倒数第二个添加到验证集，最后一个添加到测试集
            # if i % 10 == 0:
            #     test_file.append(path)
            #     test_label.append(label)
            # elif i % 10 == 9:
            #     val_file.append(path)
            #     val_label.append(label)
            # else:
            #     train_file.append(path)
            #     train_label.append(label)

            # 每10个样本中的前9个添加到训练集, 最后一个添加到测试集
            if i % 10 == 9:
                val_file.append(path)
                val_label.append(label)
            else:
                train_file.append(path)
                train_label.append(label)

        # SIMU4AE是从data.dataset引入的，具体功能是？
        # SIMU4AE返回的对象应该有scaler属性，且它可能是一个标准化器对象，包含了训练集数据的均值和标准差。
        train_dataset = SIMU4AE(train_file, train_label)
        mean, std = train_dataset.scaler.mean, train_dataset.scaler.std
        
        self.params['mean'], self.params['std'] = mean, std # 将均值和标准差存储在实验对象的参数字典中
        vali_dataset = SIMU4AE(val_file, val_label, mean, std)
        # test_dataset = SIMU4AE(test_file, test_label, mean, std)

        self.train_dataset = train_dataset
        self.vali_dataset = vali_dataset

        # print("训练集：",train_dataset.data[0])

        # 将数据集的标签转换为pytorch的tensor对象，dtype将创建的tensor对象的数据类型设置为整数类型。默认是32位整数类型。
        self.train_label = torch.tensor(train_label, dtype=torch.int).to(device)
        self.val_label = torch.tensor(val_label, dtype=torch.int).to(device)
        '''在深度学习中，通常使用整数类型的 Tensor 来表示标签或索引等离散型数据。这样做有助于减少内存占用并提高计算效率。'''

        # print("train_dataset大小:",train_dataset.data)
        # 创建用于加载各数据集的数据加载器。按照指定的batch_size对数据进行批量处理。shuffle=True表示在每个epoch开始之前对数据进行洗牌，增加训练的随机性。
        # drop_last=True 表示如果最后一个 batch 的样本数量不足一个 batch_size，就丢弃该 batch。
        self.train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
        self.vali_loader = DataLoader(vali_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
        # self.test_loader = DataLoader(test_dataset, batch_size=args.batch_size, drop_last=True)

        # print("训练dataloader：",self.train_loader.dataset)

    def next_phase(self):
        '''
        实现了模型训练的下一个阶段，并在验证集评估模型性能。
        '''
        encoder = self.model # 将当前模型（可能是编码器）存储在变量encode中
        self.model = CLS(scale_size=args.isometric_kernel, encoder=encoder).float().to(device) # 创建CLS模型实例，存储在self.model中，其参数scale_size和encoder是根据实验参数和先前模型确定。新模型被转换为浮点型并移动到指定的GPU上。
        self.train_dataset.as_day = False # 这可能是为了调整数据集的内部处理方式
        self.vali_dataset.as_day = False # as_day在SIMU4AE类中出现
        self.loss_func = nn.CrossEntropyLoss # 设置损失函数为交叉熵损失函数'nn.CrossEntropyLoss'.

        # 构建实验的设置字符串，根据该设置创建存储模型检查点的路径。
        self.setting = f"ykwcls_em{args.num_embed}-hd{args.num_hidden}-seq{args.seq_len}-cls{args.classes}" 
        _path = os.path.join(args.checkpoints, self.setting)
        if not os.path.exists(_path):
            os.makedirs(_path)

        # 打印新的模型结构，并记录可训练参数数量
        '''
        self.model.parameters(): 这部分是获取模型中所有参数的迭代器。parameters() 方法返回一个迭代器，它包含了模型中所有的参数，包括权重、偏置等。
        for p in self.model.parameters() if p.requires_grad: 这是一个生成器表达式，在模型的所有参数中筛选出那些需要进行梯度更新的参数。p.requires_grad 是一个布尔值，如果为 True，则表示该参数需要进行梯度更新，如果为 False，则表示该参数不需要进行梯度更新。
        p.numel(): 这部分是计算每个参数 p 的元素数量。numel() 方法返回张量（tensor）中元素的数量，即张量的总大小。
        sum(...): 这是对生成器表达式结果中的所有参数的元素数量进行求和操作，得到了所有可训练参数的总数量。
        '''
        print(self.model) 
        total_trainable_params = sum(
            p.numel() for p in self.model.parameters() if p.requires_grad)
        log.info(f'training parameters {total_trainable_params} ')

        # 设置一些训练超参数
        args.train_epochs = 30
        args.adjust_learning_rate = False
        args.early_stop = False
        args.learning_rate = 0.01

        # 开始训练
        self.train()
        self.model.eval() # 将模型设置为评估模式
        predicts = []
        labels = []
        # 在验证集上进行推理，获取模型对每个样本的预测，并将预测结果与真实标签进行比较，计算准确率、精确度、召回率和 F1 分数。
        
        # zip将每个样本的数据、时间戳和标签进行组合，形成一个迭代器
        for data, time_stamp, label in zip(self.vali_dataset.data, self.vali_dataset.time_stamp, self.val_label):
            l = len(data) - (len(data) % args.seq_len) # 计算当前样本的数据长度，并将其调整为可以整除序列长度的最大长度。这样做可能是为了将样本分割成等长的序列进行推理。
            x = torch.tensor(data[:l]).unsqueeze(0).reshape((-1, data.shape[1])).float().to(device) # 将当前样本的数据转换为 PyTorch 的 Tensor 对象，并调整其形状以符合模型的输入要求。
            mark = torch.tensor(time_stamp[:l]).unsqueeze(0).reshape(
                (-1, time_stamp.shape[1])).float().to(device) # 将当前样本的时间戳数据转换为 PyTorch 的 Tensor 对象，并进行形状调整和数据类型转换。
            pred = self.model(x.unsqueeze(0), mark.unsqueeze(0)) # 使用模型对当前样本进行推理，得到模型的预测结果。这里假设 self.model 是一个可以接受数据和时间戳作为输入的模型对象。
            predicted = torch.argmax(pred, dim=1).item() # 根据模型的预测结果，选取概率最大的类别作为预测值，并将其转换为 Python 标量类型。使用了 torch.argmax 函数找到概率最大的索引，然后使用 item() 方法将其转换为 Python 标量。
            # 将当前样本的预测值和真实标签分别添加到预测结果列表 predicts 和真实标签列表 labels 中，用于后续计算性能指标。
            predicts.append(predicted)
            labels.append(label.item())

        # 打印和记录模型在验证集上的性能指标
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        # 准确率：正确预测的样本数占总样本数的比例。
        print('accuracy_score', accuracy_score(labels, predicts)) # 模型在验证集上的准确率。
        # 精确度：在预测为正类别的样本中，真正为正类别的比例。多类别要对每个类别的精确度进行平均，可以是简单平均、加权平均等
        print('precision_score', precision_score(labels, predicts, average='weighted')) # 模型在验证集上的精确度。average='weighted' 表示计算加权平均精确度，即根据每个类别的样本数加权计算精确度。
        # 召回率：在所有真正为正类别的样本中，成功预测为正类别的比例。
        print('recall_score', recall_score(labels, predicts, average='weighted')) # 模型在验证集上的召回率。
        print('f1_score', f1_score(labels, predicts, average='weighted')) # 模型在验证集的F1分数，即精确度和召回率的调和平均值。

    def test_knn(self, input_data=None):
        '''
        使用knn算法对模型进行测试评估。
        '''
        self.load_model(os.path.join(self.checkpoint_path(), 'checkpoint.pth')) # 加载已训练好的模型
        self.model.eval()

        # self.knn = KNN(k=26, classes=11)
        # self.knn = KNN(k=26, classes=35)
        self.knn = KNN(k=16, classes=33, train_dataset=self.train_dataset, train_label=self.train_label)

        fusion = []
        with torch.no_grad():
            for index, (data, time_stamp) in enumerate(zip(self.train_dataset.data, self.train_dataset.time_stamp)):
                l = len(data) - (len(data) % args.seq_len) # 计算当前样本数据的长度，并将其调整为序列长度的倍数，以便后续处理。
                # x、mark分别是将样本数据和时间戳数据转换为pytorch张量，并进行形状调整，以适应模型输入的要求。
                # 转换后是三维张量：（-1,args.seq_len,data.shape[1]）
                # print("数据长度：",index,l)
                if l == 0:
                    continue  # 如果l为0，则跳过这个样本

                x = torch.tensor(data[:l]).unsqueeze(0).reshape((-1, args.seq_len, data.shape[1])).float().to(device)
                mark = torch.tensor(time_stamp[:l]).unsqueeze(0).reshape((-1, args.seq_len, time_stamp.shape[1])).float().to(device)
                # 使用模型提取特征
                '''self.model.embedding(x, mark) 将样本数据和时间戳作为输入传递给模型的嵌入层，
                然后 self.model.lgf.encoder 对嵌入后的数据进行编码。'''
                a = self.model.lgf.encoder(self.model.embedding(x, mark))
                b = [i.mean(dim=2) for i in a] # 对编码后的特征进行处理，例如计算每个特征维度的均值，以得到样本的整体特征表示。
                fusion.append(torch.stack(b, dim=2).flatten())
                # if(index == 381 or index == 472 or index== 476):
                #     # print("index: a:",a,"b:",b)
                #     print("l:",l,"len(data):",len(data),"seq_len:",args.seq_len)

            # for i, tensor in enumerate(fusion):
            #     print(f"Tensor {i} shape: {tensor.shape}")

            self.knn.fit(torch.stack(fusion), self.train_label)
            # torch.stack(fusion) 将列表 fusion 中的张量堆叠成一个张量。这个张量可能是一个二维张量，其中每一行代表一个样本，每一列代表一个特征。

            # 保存模型到文件
            self.knn.save_model('knn_model.pth')

            # 同理验证集
            vali = []
            for data, time_stamp in zip(self.vali_dataset.data, self.vali_dataset.time_stamp):
                l = len(data) - (len(data) % args.seq_len)

                if l == 0:
                    continue  # 如果l为0，则跳过这个样本

                # print("验证集数据：",data)
                # print("验证集[1]数据形状：",data.shape[1])

                x = torch.tensor(data[:l]).unsqueeze(0).reshape((-1, args.seq_len, data.shape[1])).float().to(device)
                mark = torch.tensor(time_stamp[:l]).unsqueeze(0).reshape(
                    (-1, args.seq_len, time_stamp.shape[1])).float().to(device)
                a = self.model.lgf.encoder(self.model.embedding(x, mark))
                b = [i.mean(dim=2) for i in a]
                vali.append(torch.stack(b, dim=2).flatten())

            pred = self.knn.predict(torch.stack(vali))

            if input_data is not None:
                input_vali = []
                # 从api中接收到df之后
                # print("输入数据：",input_data)
                input_time_stamp = time_features(input_data, freq=args.freq)
                
                input_bg_data = input_data['OT']
                # print("均值&标准差：",mean,std)
                scaler = StandardScaler(mean=self.params['mean'], std=self.params['std'])
                # for x in range(len(input_bg_data)):
                #     input_bg_data[x] = scaler.transform(input_bg_data[x])
                input_bg_data = scaler.transform(input_bg_data.to_numpy().reshape(-1,1))
                # print("血糖数据：",input_bg_data)



                input_l = len(input_bg_data) - (len(input_bg_data) % args.seq_len)
                # print("数据长度：",input_l)
                # print("序列长度：",args.seq_len)
                # print("数据形状：",input_bg_data.shape)

                x = torch.tensor(input_bg_data[:input_l]).unsqueeze(0).reshape((-1, args.seq_len, input_bg_data.shape[1])).float().to(device)
                mark = torch.tensor(input_time_stamp[:input_l]).unsqueeze(0).reshape(
                    (-1, args.seq_len, input_time_stamp.shape[1])).float().to(device)
                a = self.model.lgf.encoder(self.model.embedding(x, mark))
                b = [i.mean(dim=2) for i in a]
                input_vali.append(torch.stack(b, dim=2).flatten())

                # 加载模型
                loaded_model = KNN.load_model('knn_model.pth')

                # 使用加载的模型进行预测
                predictions = loaded_model.predict(torch.stack(input_vali))  # 假设 X_test 是你的测试数据
            
                return predictions
            # accuracy_class = accuracy_score(self.val_label, pred)
            # print('acc', accuracy_class)

            # 从预测中提取每个样本的第一个最可能的标签
            pred_first = pred[:, 0]
            corrects = torch.sum(pred_first == self.val_label).item()
            print('acc', corrects / len(pred_first))
            print('label', self.val_label)
            print('pred', pred)

            

    def _build_model(self):
        '''
        构建YKW模型
        '''
        model = YKW(in_features=args.in_features, seq_len=args.seq_len,
                    num_embed=args.num_embed,
                    dropout=args.dropout, freq=args.freq, device=device,
                    conv_kernel=args.conv_kernel, isometric_kernel=args.isometric_kernel).float().to(device)
        total_params = sum(p.numel() for p in model.parameters())
        log.info(f'total parameters {total_params} ') # 模型的总参数量
        total_trainable_params = sum(
            p.numel() for p in model.parameters() if p.requires_grad)
        log.info(f'training parameters {total_trainable_params} ') # 模型的可训练参数量

        if args.use_multi_gpu:
            model = nn.DataParallel(model, device_ids=args.devices) # 检查是否使用多个GPU训练
        return model

    def _select_optimizer(self):
        '''
        选择并返回一个优化器对象，用于优化模型的参数。
        使用adam优化算法实例化一个优化器对象
        Adam优化算法是一种常用的梯度下降算法的变体，结合了动量法和自适应学习率的特性。
        '''
        model_optim = optim.Adam(self.model.parameters(), lr=args.learning_rate) # 两个参数分别是模型的参数和学习率
        return model_optim

    def _loss_function(self, pred, true):
        '''
        定义损失函数的计算方式。
        '''
        criterion = self.loss_func() # 获取损失函数的实例
        return criterion(pred, true) # 计算预测值和真实值的损失并返回

    def _train_loader(self):
        return self.train_loader # 返回训练数据加载器对象

    def _vali_loader(self):
        return self.vali_loader # 返回验证数据加载器对象

    def _test_loader(self):
        return self.test_loader # 返回测试数据加载器对象

