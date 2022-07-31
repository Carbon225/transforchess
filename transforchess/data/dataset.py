import datasets

from transforchess.paths import DATASET


def load_dataset():
    return datasets.load_dataset('csv', 'chess', data_files=[DATASET], split='train')
