from chess import Board


class HumanAgent:
    def make_move(self, board: Board):
        while True:
            try:
                move = input('Enter your move: ')
                board.push_san(move)
                break
            except ValueError:
                print('Invalid move')
