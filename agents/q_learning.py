import random
import numpy as np
from agents.base import BaseAgent
from utils.io_utils import save_q_table, load_q_table

class QLearningAgent(BaseAgent):
    def __init__(self, name, epsilon=0.1, alpha=0.1, gamma=0.9):
        self.name = name
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma
        self.q_table = {}
        self.states = []

    def make_move(self, board):
        valid_moves = [i for i, val in enumerate(board) if val == " "]
        state = tuple(board)

        if state not in self.q_table:
            self.q_table[state] = [0 for _ in range(9)]

        if random.random() < self.epsilon:
            move = random.choice(valid_moves)
        else:
            q_values = [self.q_table[state][i] if i in valid_moves else -float('inf') for i in range(9)]
            move = int(np.argmax(q_values))

        self.states.append((state, move))
        return move

    def learn(self, reward):
        for state, action in self.states:
            max_q = max(self.q_table[state])
            self.q_table[state][action] += self.alpha * (reward + self.gamma * max_q - self.q_table[state][action])
        self.states.clear()

    def save(self, path="checkpoints/q_table_{}.json"):
        save_q_table(self.q_table, path.format(self.name))

    def load(self, path="checkpoints/q_table_{}.json"):
        self.q_table = load_q_table(path.format(self.name))

    def clone_eval_agent(self):
        agent = QLearningAgent(self.name + "_eval", epsilon=0.0, alpha=self.alpha, gamma=self.gamma)
        agent.q_table = self.q_table.copy()
        return agent