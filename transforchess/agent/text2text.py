from transformers import pipeline
from chess import Board, WHITE
import regex as re
from random import choice

from transforchess.parser import san2human, human2san
from transforchess.model.config import MODEL_CHECKPOINT


MODEL_MOVE_REGEX = re.compile(r'^ (?:White|Black) ([^.]+)\.$')


class Text2TextAgent:
    def __init__(self):
        self.task = pipeline(
            'text2text-generation',
            MODEL_CHECKPOINT,
        )

        self.game = ''

    def make_move(self, board: Board):
        if len(board.move_stack) > 0:
            move = board.pop()
            player = 'White' if board.turn == WHITE else 'Black'
            self.game += f' {player} {san2human(board.san(move))}.'
            board.push(move)

        sequences = self.task(self.game, num_return_sequences=4)

        for sequence in sequences:
            try:
                move_human = MODEL_MOVE_REGEX \
                    .match(sequence['generated_text']) \
                    .group(1)
                move_san = human2san(move_human)
                board.push_san(move_san)
                move = board.pop()
                break
            except Exception as e:
                print('Error:', e)
        else:
            print('Choosing random move...')
            move = choice(list(board.legal_moves))

        player = 'White' if board.turn == WHITE else 'Black'
        self.game += f' {player} {san2human(board.san(move))}.'
        board.push(move)
