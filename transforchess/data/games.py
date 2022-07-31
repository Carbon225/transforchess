import regex as re
from random import shuffle
from tqdm import tqdm

from transforchess.parser import san2human
from transforchess.paths import GAMES, DATASET


GAME_REGEX = re.compile(r'(^1\.(?:.+\n?)+$)', re.MULTILINE)
TURN_REGEX = re.compile(r'\d+\.')
GAME_RESULT_REGEX = re.compile(r'^(?:1-0|0-1|1\/2-1\/2|\*)$')


def parse_game(game: str) -> str:
    game = game.strip()

    out = ''

    turns = TURN_REGEX.split(game)

    for turn in turns:
        if len(turn) == 0:
            continue
        moves = turn.strip().split()
        player = 'White'
        for move in moves:
            if GAME_RESULT_REGEX.match(move):
                out += f',{move}'
                break
            out += f' {player} {san2human(move)}.'
            player = 'Black' if player == 'White' else 'White'

    return out


def make_dataset(num_games: int = None):
    games = []
    
    with open(GAMES, 'r') as f:
        games = GAME_REGEX.findall(f.read())
        
    shuffle(games)
    
    if num_games is not None:
        games = games[:num_games]

    with open(DATASET, 'w') as f:
        f.write('text,label\n')
        for game in tqdm(games):
            line = parse_game(game).strip()
            f.write(line + '\n')
