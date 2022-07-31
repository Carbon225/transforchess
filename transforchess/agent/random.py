from chess import Board
from random import choice


class RandomAgent:
    def make_move(self, board: Board):
        move = choice(list(board.legal_moves))
        board.push(move)
