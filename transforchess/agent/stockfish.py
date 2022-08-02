from transforchess.paths import STOCKFISH
from chess.engine import SimpleEngine, Limit
from chess import Board


class StockfishAgent:
    def make_move(self, board: Board):
        engine = SimpleEngine.popen_uci(STOCKFISH)
        engine.configure({'Skill Level': 1})
        try:
            move = engine.play(board, Limit(time=0.001)).move
        finally:
            engine.quit()
        board.push(move)
