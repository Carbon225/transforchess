import numpy as np
from chess import Board
from transformers import pipeline

from transforchess.parser import human2san


class GameLength:
    def __init__(self, method: str, n_games: int):
        self.n_games = n_games
        self.method = {
            'median': np.median,
            'mean': np.mean,
        }[method]

    def __call__(self, model, tokenizer) -> int:
        task = pipeline('text2text-generation', model=model, tokenizer=tokenizer)
        lengths = [self._run_game(task) for _ in range(self.n_games)]
        return self.method(lengths)

    def _run_game(self, task) -> int:
        board = Board()
        game = ''
        while not board.is_game_over():
            try:
                move_human = task(game)[0]['generated_text']
                move_san = human2san(move_human[7:-1])
                board.push_san(move_san)
                game += move_human
            except Exception:
                break
        return len(board.move_stack)
