import argparse

from transforchess.model import pretrain


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()

    pretrain(args.resume)
