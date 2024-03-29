from config.args import device
import torch


class KNN:
    def __init__(self, k, classes):
        self.weights = None
        self.y_train = None
        self.X_train = None
        self.k = k
        self.classes = classes

    def fit(self, X, y, weights=[]):
        self.X_train = X.to(device)
        self.y_train = y.to(device)
        if not weights:
            self.weights = [1] * len(self.classes)
        else:
            self.weights = weights

    def predict(self, X):
        y_pred = [self._predict(x) for x in X]
        return torch.tensor(y_pred).to(device)

    def _predict(self, x):
        # 计算x与所有训练样本的欧氏距离
        distances = [torch.norm(x - x_train) for x_train in self.X_train]
        # 找到K个最近邻居的索引
        k_indices = torch.topk(torch.tensor(distances), self.k, largest=False).indices
        # 获取K个最近邻居的标签
        k_nearest_labels = [self.y_train[i] for i in k_indices]
        # 多数表决法来预测标签
        most_common = torch.bincount(torch.tensor(k_nearest_labels)).argmax()

        return most_common.item()

        # def calculate_weight(neighbour_point_indexs):
        #     weights = []
        #     for index in neighbour_point_indexs:
        #         for idx, l in enumerate(literals, start=0):
        #             if self.train_label[index] == l:
        #                 weights.append(math.sqrt(self.weights[idx]))
        #     return weights
        #
        # def calculate_total_weight(distances):
        #     res = []
        #     for j in range(len(distances)):
        #         neighbour_point_indexs = knn.kneighbors([data_test[j]], K, False)
        #         weights = calculate_weight(neighbour_point_indexs[0])
        #         res.append(weights)
        #     return np.array(res)
