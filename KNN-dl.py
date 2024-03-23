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


def predict(target):
    final_data = pd.DataFrame(pd.read_csv('datasets/final_data.csv', encoding='ISO-8859-1')).replace('â\\x80?', '-')
    print(final_data.columns.values)
    data = final_data.drop(labels='recommendation', axis=1).to_numpy()
    labels = final_data.loc[:, 'recommendation'].to_numpy()

    # 划分训练集和测试集
    data_train, data_test, labels_train, labels_test = train_test_split(data, labels, test_size=0.2, random_state=4)

    K = 26

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

    def calculate_total_weight(distances):
        res = []
        for j in range(len(distances)):
            neighbour_point_indexs = knn.kneighbors([data_test[j]], K, False)
            weights = calculate_weight(neighbour_point_indexs[0])
            res.append(weights)
        return np.array(res)

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
