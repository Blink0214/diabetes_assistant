from config.args import device
from sklearn.neighbors import KNeighborsClassifier
import torch

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

class KNN:
    def __init__(self, k, classes, train_dataset=None, train_label=None):
        self.weights = None
        self.y_train = None
        self.X_train = None
        self.k = k
        self.classes = classes

        # label_counts = [1.0] * len(literals)
        # self.weights = [1] * len(literals)
        # all_count = 1.0

        # for recommendation in train_label:
        #     for idx, l in enumerate(literals, start=0):
        #         if recommendation == l:
        #             label_counts[idx] += 1
        #     all_count += 1
        # count = 1
        # for i in range(len(self.weights)):
        #     self.weights[i] = count / label_counts[i]
        # for recommendation in train_dataset.labels:
        #     for idx, l in enumerate(literals, start=0):
        #         if recommendation == l:
        #             label_counts[idx] += 1
        #     all_count += 1
        # count = 1
        # for i in range(len(self.weights)):
        #     self.weights[i] = count / label_counts[i]

        # 加上权重处理(单独)
        # self.knn = KNeighborsClassifier(n_neighbors=self.classes, weights=self.weights)
        # self.knn = KNeighborsClassifier(n_neighbors=self.classes, weights='uniform')

    def fit(self, X, y, weights=[]):
        # 用于训练 KNN 模型。接收训练数据 X 和对应的标签 y，以及可选的权重列表 weights。
        self.X_train = X.to(device)
        self.y_train = y.to(device)

        if not weights:
            self.weights = [1] * self.classes
        else:
            self.weights = weights
        # self.knn.fit(X=X,y=y)
        

    def predict(self, X):
        # 对给定的输入数据 X 进行预测。
        y_pred = [self._predict(x) for x in X]
        return torch.tensor(y_pred).to(device)

        return self.knn.predict(X=X)

    def _predict(self, x):
        # 计算x与所有训练样本的欧氏距离
        distances = [torch.norm(x - x_train) for x_train in self.X_train]
        # 找到K个最近邻居的索引
        k_indices = torch.topk(torch.tensor(distances), self.k, largest=False).indices
        # 获取K个最近邻居的标签
        k_nearest_labels = [self.y_train[i] for i in k_indices]

        # 统计每个标签的票数
        label_counts = torch.bincount(torch.tensor(k_nearest_labels))
        
        # 找到前3个最高票数的标签
        top_three_labels_indices = torch.argsort(label_counts, descending=True)[:3]
        top_three_labels = top_three_labels_indices.tolist()
        
        return top_three_labels

        # 多数表决法来预测标签      
        most_common = torch.bincount(torch.tensor(k_nearest_labels)).argmax()
        return most_common.item()
    
    def save_model(self, filename):
        # 保存模型参数到文件
        torch.save({
            'k': self.k,
            'classes': self.classes,
            'weights': self.weights,
            'X_train': self.X_train,
            'y_train': self.y_train
        }, filename)

    @classmethod
    def load_model(cls, filename):
        # 从文件中加载模型参数并重新构建模型
        checkpoint = torch.load(filename)
        model = cls(k=checkpoint['k'], classes=checkpoint['classes'])
        model.weights = checkpoint['weights']
        model.X_train = checkpoint['X_train']
        model.y_train = checkpoint['y_train']
        return model

        def calculate_weight(self,neighbour_point_indexs):
            weights = []
            for index in neighbour_point_indexs:
                for idx, l in enumerate(literals, start=0):
                    if self.train_label[index] == l:
                        weights.append(math.sqrt(self.weights[idx]))
            return weights

        def calculate_total_weight(distances):
            res = []
            for j in range(len(distances)):
                neighbour_point_indexs = knn.kneighbors([data_test[j]], K, False)
                weights = calculate_weight(neighbour_point_indexs[0])
                res.append(weights)
            return np.array(res)
