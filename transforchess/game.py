from chess import Board


class Game:
    def __init__(self, white, black):
        self.white = white
        self.black = black
        self.board = Board()

    def get_last_move_san(self) -> str:
        move = self.board.pop()
        san = self.board.san(move)
        self.board.push(move)
        return san

    def play(self) -> str:
        try:
            turn = 1
            while True:
                self.white.make_move(self.board)
                print(f'{turn}. {self.get_last_move_san()}')

                if self.board.is_game_over():
                    break

                self.black.make_move(self.board)
                print(f'{" " * len(str(turn))}  {self.get_last_move_san()}')

                if self.board.is_game_over():
                    break

                print(self.board)

                turn += 1

        except Exception as e:
            print(e)

        finally:
            return self.board.result()
