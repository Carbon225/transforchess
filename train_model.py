import argparse

from transforchess.model.gpt2clm import train


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()

    train(args.resume)
