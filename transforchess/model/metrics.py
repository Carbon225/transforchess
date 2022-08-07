import numpy as np
from random import choice
from chess import Board
from transformers import pipeline

from transforchess.parser import human2san, san2human


class GameLength:
    def __init__(self, method: str, n_games: int):
        self.n_games = n_games
        self.method = {
            'median': np.median,
            'mean': np.mean,
        }[method]

    def __call__(self, model, tokenizer, device) -> float:
        task = pipeline(
            'text2text-generation',
            model=model,
            tokenizer=tokenizer,
            device=device,
        )

        boards = [Board() for _ in range(self.n_games)]
        games = ['' for _ in range(self.n_games)]
        is_over = [False] * self.n_games

        while not all(is_over):
            inds = [i for i, over in enumerate(is_over) if not over]

            sequences = task([games[i] for i in inds])

            for i, sequence in enumerate(sequences):
                try:
                    move_human = sequence['generated_text']
                    move_san = human2san(move_human[7:-1])
                    boards[inds[i]].push_san(move_san)
                    games[inds[i]] += move_human

                    move = choice(list(boards[inds[i]].legal_moves))
                    san = boards[inds[i]].san(move)
                    boards[inds[i]].push(move)
                    games[inds[i]] += f' Black {san2human(san)}.'

                    if boards[inds[i]].is_game_over():
                        is_over[inds[i]] = True
                except Exception:
                    is_over[inds[i]] = True

        return self.method([len(board.move_stack) for board in boards])
