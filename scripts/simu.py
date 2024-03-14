import os

from experiments.expsimu import Expsimu


def adolescent():
    exp = Expsimu(seed=2024, subject='adolescent', in_features=1, out_features=11, dataset_dir=os.path.join('.', 'datasets', 'dl-TrainSet'))
    exp.train()