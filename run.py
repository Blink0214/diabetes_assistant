from scripts.simu import adolescent, rnn

exps = {
    'adolescent': adolescent,
    'rnn': rnn,
}

exps['rnn']()
