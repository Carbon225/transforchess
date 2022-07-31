import argparse
from transforchess.game import Game
from transforchess.agent import (
    RandomAgent,
    StockfishAgent,
    BartAgent,
)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('white', type=str)
    parser.add_argument('black', type=str)
    args = parser.parse_args()

    if args.white == 'random':
        white = RandomAgent()
    elif args.white == 'stockfish':
        white = StockfishAgent()
    elif args.white == 'bart':
        white = BartAgent()
    else:
        raise ValueError(f'Unknown white agent: {args.white}')

    if args.black == 'random':
        black = RandomAgent()
    elif args.black == 'stockfish':
        black = StockfishAgent()
    elif args.black == 'bart':
        black = BartAgent()
    else:
        raise ValueError(f'Unknown black agent: {args.black}')

    game = Game(white, black)

    result = game.play()
    if result == '1-0':
        print('Winner:', args.white, 'white')
    elif result == '0-1':
        print('Winner:', args.black, 'black')
    else:
        print(result)
