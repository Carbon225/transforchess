from transformers import pipeline
from random import choice
from chess import Board


class GPT2Agent:
    def __init__(self):
        self.task = pipeline(
            'text-generation',
            'raruidol/GameANchess',
        )

        self.task.model.config.pad_token_id = self.task.model.config.eos_token_id

        self.game = ''

    def make_move(self, board: Board):
        if len(board.move_stack) > 0:
            move = board.pop()
            self.game += f'{board.san(move)} '
            board.push(move)

        try:
            sequence = self.task(self.game.strip(), max_new_tokens=10)[0]
            move_san = sequence['generated_text'] \
                [len(self.game):] \
                .split()[0]
            board.push_san(move_san)
            move = board.pop()
        except Exception as e:
            print('Error:', e)
            print('Choosing random move...')
            move = choice(list(board.legal_moves))

        self.game += f'{board.san(move)} '
        board.push(move)
