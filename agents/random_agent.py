import random
from agents.base import BaseAgent

class RandomAgent(BaseAgent):
    def __init__(self, name):
        self.name = name

    def make_move(self, board):
        valid_moves = [i for i, val in enumerate(board) if val == " "]
        return random.choice(valid_moves)
