import argparse
from transforchess.game import Game
from transforchess.agent import (
    RandomAgent,
    StockfishAgent,
    BartAgent,
    GPT2Agent,
    HumanAgent,
)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('white', choices=['random', 'stockfish', 'bart', 'gpt2', 'human'])
    parser.add_argument('black', choices=['random', 'stockfish', 'bart', 'gpt2', 'human'])
    args = parser.parse_args()

    agents = {
        'random': RandomAgent,
        'stockfish': StockfishAgent,
        'bart': BartAgent,
        'gpt2': GPT2Agent,
        'human': HumanAgent,
    }
    
    game = Game(agents[args.white](), agents[args.black]())

    result = game.play()
    if result == '1-0':
        print('Winner:', args.white, 'white')
    elif result == '0-1':
        print('Winner:', args.black, 'black')
    else:
        print(result)
