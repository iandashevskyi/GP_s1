import random
from agents.base import BaseAgent

class RuleBasedAgent(BaseAgent):
    def __init__(self, name):
        self.name = name

    def make_move(self, board):
        symbol = self._infer_symbol(board)
        return self.rule_based_move(board, symbol)

    def rule_based_move(self, board, player_symbol):
        opponent_symbol = "X" if player_symbol == "O" else "O"

        for move in self.get_empty_cells(board):
            board[move] = player_symbol
            if self.check_win(board, player_symbol):
                board[move] = " "
                return move
            board[move] = " "

        for move in self.get_empty_cells(board):
            board[move] = opponent_symbol
            if self.check_win(board, opponent_symbol):
                board[move] = " "
                return move
            board[move] = " "

        if board[4] == " ":
            return 4

        corners = [0, 2, 6, 8]
        empty_corners = [c for c in corners if board[c] == " "]
        if empty_corners:
            return random.choice(empty_corners)

        sides = [1, 3, 5, 7]
        empty_sides = [s for s in sides if board[s] == " "]
        if empty_sides:
            return random.choice(empty_sides)

        return random.choice(self.get_empty_cells(board))

    def get_empty_cells(self, board):
        return [i for i, cell in enumerate(board) if cell == " "]

    def check_win(self, board, symbol):
        win_conditions = [
            (0, 1, 2), (3, 4, 5), (6, 7, 8),
            (0, 3, 6), (1, 4, 7), (2, 5, 8),
            (0, 4, 8), (2, 4, 6)
        ]
        for i, j, k in win_conditions:
            if board[i] == board[j] == board[k] == symbol:
                return True
        return False

    def _infer_symbol(self, board):
        x_count = board.count("X")
        o_count = board.count("O")
        return "X" if x_count == o_count else "O"
