from transformers import pipeline
import torch
import chess
import chess.engine
import random
import regex as re

from transforchess.parser import san2human, human2san
from transforchess.model.bartqa.paths import MODEL
from transforchess.paths import STOCKFISH


MODEL_MOVE_REGEX = re.compile(r'^ (?:White|Black) ([^.]+)\.$')


class Game:
    def __init__(self, white: str, black: str):
        device = torch.device('cpu')
        
        self.task = pipeline(
            'text2text-generation',
            MODEL,
            device=device,
        )

        if white == 'stockfish':
            self.get_white_move = self.get_engine_move
        elif white == 'transformer':
            self.get_white_move = self.get_transformer_move
        elif white == 'random':
            self.get_white_move = self.get_random_move
        else:
            raise ValueError(f'Unknown player: {white}')

        if black == 'stockfish':
            self.get_black_move = self.get_engine_move
        elif black == 'transformer':
            self.get_black_move = self.get_transformer_move
        elif black == 'random':
            self.get_black_move = self.get_random_move
        else:
            raise ValueError(f'Unknown player: {black}')

        self.engine = chess.engine.SimpleEngine.popen_uci(STOCKFISH)
        self.board = chess.Board()
        self.game = ''

    def get_random_move(self) -> chess.Move:
        return random.choice(list(self.board.legal_moves))

    def get_engine_move(self) -> chess.Move:
        return self.engine.play(self.board, chess.engine.Limit(time=0.001)).move

    def get_transformer_move(self) -> chess.Move:
        sequences = self.task(self.game, num_return_sequences=4)

        for sequence in sequences:
            try:
                move_human = MODEL_MOVE_REGEX \
                    .match(sequence['generated_text']) \
                    .group(1)
                move_san = human2san(move_human)
                self.board.push_san(move_san)
                move = self.board.pop()
                if self.board.san(move) != move_san:
                    raise ValueError()
                return move
            except ValueError:
                print('Error making move, trying again...')
        else:
            print('Error making move, choosing random...')
            return self.get_random_move()

    def push_white_move(self, move: chess.Move):
        self.game += f' White {san2human(self.board.san(move))}.'
        self.board.push(move)

    def push_black_move(self, move: chess.Move):
        self.game += f' Black {san2human(self.board.san(move))}.'
        self.board.push(move)

    def get_white_move(self) -> chess.Move:
        return self.get_random_move()

    def get_black_move(self) -> chess.Move:
        return self.get_random_move()

    def play(self) -> str:
        try:
            turn = 1
            while True:
                move = self.get_white_move()
                print(f'{turn}. {self.board.san(move)}')
                self.push_white_move(move)

                if self.board.is_game_over():
                    break

                move = self.get_black_move()
                print(f'{" " * len(str(turn))}  {self.board.san(move)}')
                self.push_black_move(move)

                if self.board.is_game_over():
                    break

                print(self.board)
                print('.'.join(self.game.split('.')[-5:]).strip())

                turn += 1

        except Exception as e:
            print(e)

        finally:
            self.engine.quit()
            return self.board.result()
