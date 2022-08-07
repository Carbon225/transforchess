import argparse

from transforchess.data import make_dataset


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--count', type=int, default=None)
    args = parser.parse_args()
    make_dataset(args.count)
