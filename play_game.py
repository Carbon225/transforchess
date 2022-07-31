import argparse
from transforchess.game import Game


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('white', type=str)
    parser.add_argument('black', type=str)
    args = parser.parse_args()

    game = Game(args.white, args.black)

    result = game.play()
    if result == '1-0':
        print('Winner:', args.white, 'white')
    elif result == '0-1':
        print('Winner:', args.black, 'black')
    else:
        print(result)
