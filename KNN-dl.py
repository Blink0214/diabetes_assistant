import math
import random

from scipy.constants import sigma
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


def add_data(line):
    final_data = pd.DataFrame(pd.read_csv('datasets/final_data.csv', encoding='ISO-8859-1'))
    final_data = final_data.append(line, ignore_index=True)
    final_data.to_csv('datasets/final_data.csv')
    # 读取final_data.csv数据集，添加新的数据行，并再保存回final_data.csv
    # ignore_index=True表明追加数据时忽略现有索引，生成新的连续索引


def predict(target):
    final_data = pd.DataFrame(pd.read_csv('datasets/final_data.csv', encoding='ISO-8859-1')).replace('â\\x80?', '-')
    print(final_data.columns.values)
    data = final_data.drop(labels='recommendation', axis=1).to_numpy()
    labels = final_data.loc[:, 'recommendation'].to_numpy()
    # data是删除recommendation列后的数据，并从DataFrame转换为Numpy数组。
    # labels是只获取了recommendation列的数据，并将DataFrame转换为Numpy数组。

    # 划分训练集和测试集
    data_train, data_test, labels_train, labels_test = train_test_split(data, labels, test_size=0.2, random_state=4)

    K = 26
    # 选择距离样本最近的26个邻居样本，根据他们的类别决定该样本的类别。


    # 统计训练集中各类别样本的数量，用于计算权重，避免样本数量不平衡导致模型偏倚。
    # 初始值均为1.0，避免有样本数量为0导致除数错误.count用于计算总体样本数量。
    # 平滑项（1）：Laplace平滑（Laplace Smoothing）或添加平滑（Addditive Smoothing）
    count_increase_basal = 1.0
    count_decrease_basal = 1.0
    count_no_adjustment_recommended = 1.0
    count_increase_am_bolus = 1.0
    count_increase_pm_bolus = 1.0
    count_increase_evening_bolus = 1.0
    count_increase_overnight_bolus = 1.0
    count_decrease_am_bolus = 1.0
    count_decrease_pm_bolus = 1.0
    count_decrease_evening_bolus = 1.0
    count_decrease_overnight_bolus = 1.0
    count = 1.0
    for recommendation in labels_train:
        if recommendation == 'increase_basal':
            count_increase_basal = count_increase_basal + 1
        elif recommendation == 'decrease_basal':
            count_decrease_basal = count_decrease_basal + 1
        elif recommendation == 'no_adjustment_recommended':
            count_no_adjustment_recommended = count_no_adjustment_recommended + 1

        elif recommendation == 'increase AM bolus':
            count_increase_am_bolus = count_increase_am_bolus + 1
        elif recommendation == 'decrease AM bolus':
            count_decrease_am_bolus = count_decrease_am_bolus + 1

        elif recommendation == 'increase PM bolus':
            count_increase_pm_bolus = count_increase_pm_bolus + 1
        elif recommendation == 'decrease PM bolus':
            count_decrease_pm_bolus = count_decrease_pm_bolus + 1

        elif recommendation == 'increase evening bolus':
            count_increase_evening_bolus = count_increase_evening_bolus + 1
        elif recommendation == 'decrease evening bolus':
            count_decrease_evening_bolus = count_decrease_evening_bolus + 1

        elif recommendation == 'increase overnight bolus':
            count_increase_overnight_bolus = count_increase_overnight_bolus + 1
        elif recommendation == 'decrease overnight bolus':
            count_decrease_overnight_bolus = count_decrease_overnight_bolus + 1
        count += 1

    # 权重计算公式: weight=1/(count+1)
    count = 1
    weight_increase_basal = count / count_increase_basal
    weight_decrease_basal = count / count_decrease_basal
    weight_no_adjustment_recommended = count / count_no_adjustment_recommended
    weight_increase_am_bolus = count / count_increase_am_bolus
    weight_increase_pm_bolus = count / count_increase_pm_bolus
    weight_increase_evening_bolus = count / count_increase_evening_bolus
    weight_increase_overnight_bolus = count / count_increase_overnight_bolus
    weight_decrease_am_bolus = count / count_decrease_am_bolus
    weight_decrease_pm_bolus = count / count_decrease_pm_bolus
    weight_decrease_evening_bolus = count / count_decrease_evening_bolus
    weight_decrease_overnight_bolus = count / count_decrease_overnight_bolus


    # 将预测结果中的字符串类别标签转换为数字类别标签，并存储在nums列表中返回
    def get_num(predicts):
        nums = []
        for s in predicts:
            if s == 'increase_basal':
                nums.append(0)
            elif s == 'decrease_basal':
                nums.append(1)
            elif s == 'no_adjustment_recommended':
                nums.append(2)
            elif s == 'increase AM bolus':
                nums.append(3)
            elif s == 'decrease AM bolus':
                nums.append(4)

            elif s == 'increase PM bolus':
                nums.append(5)
            elif s == 'decrease PM bolus':
                nums.append(6)

            elif s == 'increase evening bolus':
                nums.append(7)
            elif s == 'decrease evening bolus':
                nums.append(8)

            elif s == 'increase overnight bolus':
                nums.append(9)
            elif s == 'decrease overnight bolus':
                nums.append(10)
        return nums


    # 对单个样本点的所有邻居点计算权重并存储在列表中。
    # 根据邻居点的类别标签从预先计算好的权重中选择合适的权重，并将这些权重计算为平方根后存储在列表中并返回。
    # neighbour_point_indexs是包含了所有邻居点索引的列表。
    # 该函数遍历邻居点索引列表，获取每个邻居点的类别标签，并根据标签获取相应的权重值（存在temp中），将temp的平方根存储在weights列表中并返回。
    def calculate_weight(neighbour_point_indexs):
        weights = []
        for index in neighbour_point_indexs:
            temp = 0
            if labels_train[index] == 'increase_basal':
                temp = weight_increase_basal
            elif labels_train[index] == 'decrease_basal':
                temp = weight_decrease_basal
            elif labels_train[index] == 'no_adjustment_recommended':
                temp = weight_no_adjustment_recommended

            elif labels_train[index] == 'increase AM bolus':
                temp = weight_increase_am_bolus
            elif labels_train[index] == 'decrease AM bolus':
                temp = weight_decrease_am_bolus

            elif labels_train[index] == 'increase PM bolus':
                temp = weight_increase_pm_bolus
            elif labels_train[index] == 'decrease PM bolus':
                temp = weight_decrease_pm_bolus

            elif labels_train[index] == 'increase evening bolus':
                temp = weight_increase_evening_bolus
            elif labels_train[index] == 'decrease evening bolus':
                temp = weight_decrease_evening_bolus

            elif labels_train[index] == 'increase overnight bolus':
                temp = weight_increase_overnight_bolus
            elif labels_train[index] == 'decrease overnight bolus':
                temp = weight_decrease_overnight_bolus
            weights.append(math.sqrt(temp))
        return weights


    # 计算每个测试样本的邻居点的总体权重，并存储在数组中返回
    # distances包含测试样本到期最近邻居点的距离的列表数组。
    ''' 对每个测试样本的操作：
        找到该测试样本的最近的26个邻居点的索引。
        将这些邻居点的索引传递给 calculate_weight() 函数，以计算邻居点的权重。
        将计算得到的权重存储在 res 列表中。
    '''
    def calculate_total_weight(distances):
        res = []
        for j in range(len(distances)):
            neighbour_point_indexs = knn.kneighbors([data_test[j]], K, False) # knn.kneighbors接收数组维度为（n_samples,n_features），n_samples是数据点数量，n_features是每个数据点的特征数。需要传入二维数组。
            weights = calculate_weight(neighbour_point_indexs[0])
            res.append(weights)
        return np.array(res)

    '''
        指定如何对邻居点进行加权。使用 lambda 函数，该函数接受距离数组作为输入参数，并调用 calculate_total_weight() 函数来计算邻居点的总体权重。
        这种方式允许根据邻居点的距离和权重来动态地调整预测过程中的邻居点的重要性，以提高模型性能。
    '''
    knn = KNeighborsClassifier(n_neighbors=K, weights=lambda distances: calculate_total_weight(distances))
    knn.fit(data_train, labels_train)
    labels_predict = knn.predict(data_test)

    # 计算预测准确率
    accuracy = accuracy_score(labels_test, labels_predict)
    print('predict', set(get_num(labels_predict)))
    print('label', set(get_num(labels_test)))
    return accuracy


# 计算测试集上的预测正确率
test_accuracy = predict([])
print("Test Set Accuracy:", test_accuracy)
