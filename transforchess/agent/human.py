from chess import Board


class HumanAgent:
    def make_move(self, board: Board):
        move = input('Enter your move: ')
        board.push_san(move)
