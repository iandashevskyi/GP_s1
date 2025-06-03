import random
from agents.base import BaseAgent

class LookaheadAgent(BaseAgent):
    def __init__(self, name):
        self.name = name

    def make_move(self, board):
        available = [i for i, val in enumerate(board) if val == " "]
        symbol = self._infer_symbol(board)
        # Check for winning move
        for move in available:
            if self.check_winning_move(board, move, symbol):
                return move
        # Check for blocking opponent
        opponent_symbol = "O" if symbol == "X" else "X"
        for move in available:
            if self.check_winning_move(board, move, opponent_symbol):
                return move
        # Else, random move
        return random.choice(available)

    def check_winning_move(self, board, move, symbol):
        temp_board = board.copy()
        temp_board[move] = symbol
        win_conditions = [
            (0, 1, 2), (3, 4, 5), (6, 7, 8),
            (0, 3, 6), (1, 4, 7), (2, 5, 8),
            (0, 4, 8), (2, 4, 6)
        ]
        for i, j, k in win_conditions:
            if temp_board[i] == temp_board[j] == temp_board[k] == symbol:
                return True
        return False

    def _infer_symbol(self, board):
        x_count = board.count("X")
        o_count = board.count("O")
        return "X" if x_count == o_count else "O"
