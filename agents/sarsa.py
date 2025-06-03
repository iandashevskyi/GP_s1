import random
import numpy as np
from agents.base import BaseAgent
from utils.io_utils import save_q_table, load_q_table

class SarsaAgent(BaseAgent):
    def __init__(self, name, epsilon=0.1, alpha=0.1, gamma=0.7):
        self.name = name
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma
        self.q_table = {}
        self.last_state = None
        self.last_action = None

    def make_move(self, board):
        valid_moves = [i for i, val in enumerate(board) if val == " "]
        state = tuple(board)

        if state not in self.q_table:
            self.q_table[state] = [0 for _ in range(9)]

        if random.random() < self.epsilon:
            action = random.choice(valid_moves)
        else:
            q_values = [self.q_table[state][i] if i in valid_moves else -float('inf') for i in range(9)]
            action = int(np.argmax(q_values))

        self.last_state = state
        self.last_action = action
        return action

    def learn(self, reward, next_state, next_action):
        if self.last_state is None or self.last_action is None:
            return

        if next_state is None or next_action is None:
            td_target = reward  # final state
        else:
            if next_state not in self.q_table:
                self.q_table[next_state] = [0 for _ in range(9)]
            next_q = self.q_table[next_state][next_action]
            td_target = reward + self.gamma * next_q

        current_q = self.q_table[self.last_state][self.last_action]
        self.q_table[self.last_state][self.last_action] += self.alpha * (td_target - current_q)

        if next_state is not None and next_action is not None:
            self.last_state = next_state
            self.last_action = next_action
        else:
            self.last_state = None
            self.last_action = None


    def reset(self):
        self.last_state = None
        self.last_action = None

    def save(self, path):
        save_q_table(self.q_table, path)

    def load(self, path):
        self.q_table = load_q_table(path)

    def clone_eval_agent(self):
        agent = SarsaAgent(self.name + "_eval", epsilon=0.0, alpha=self.alpha, gamma=self.gamma)
        agent.q_table = self.q_table.copy()
        return agent