from scripts.simu import adolescent, rnn, knn

exps = {
    'adolescent': adolescent,
    'rnn': rnn,
    'knn': knn,
}

exps['adolescent']()

'''
导入adolescent、rnn、knn三个实验，并存储在一个名为exps的字典中。
然后，调用adolescent实验函数，执行该实验。
'''