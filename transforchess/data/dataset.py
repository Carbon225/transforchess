import datasets


def load_dataset():
    return datasets.load_dataset('carbon225/lichess-elite', split='train')
