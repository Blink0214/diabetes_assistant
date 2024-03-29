from scripts.simu import adolescent, rnn, knn

exps = {
    'adolescent': adolescent,
    'rnn': rnn,
    'knn': knn,
}

exps['adolescent']()
