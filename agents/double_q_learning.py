import random
import numpy as np
from agents.base import BaseAgent

class DoubleQLearningAgent(BaseAgent):
    def __init__(self, name, epsilon=0.1, alpha=0.1, gamma=0.9):
        self.name = name
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma
        self.q1 = {}
        self.q2 = {}
        self.states = []

    def make_move(self, board):
        valid_moves = [i for i, val in enumerate(board) if val == " "]
        state = tuple(board)

        if state not in self.q1:
            self.q1[state] = [0 for _ in range(9)]
        if state not in self.q2:
            self.q2[state] = [0 for _ in range(9)]

        if random.random() < self.epsilon:
            move = random.choice(valid_moves)
        else:
            q_sum = [self.q1[state][i] + self.q2[state][i] if i in valid_moves else -float('inf') for i in range(9)]
            move = int(np.argmax(q_sum))

        self.states.append((state, move))
        return move

    def learn(self, reward):
        for state, action in self.states:
            if random.random() < 0.5:
                next_q = self.q2
                update_q = self.q1
            else:
                next_q = self.q1
                update_q = self.q2

            if state not in next_q:
                next_q[state] = [0 for _ in range(9)]
            if state not in update_q:
                update_q[state] = [0 for _ in range(9)]

            max_action = int(np.argmax(next_q[state]))
            target = reward + self.gamma * next_q[state][max_action]
            update_q[state][action] += self.alpha * (target - update_q[state][action])

        self.states.clear()

    def save(self, path):
        import json, os
        os.makedirs(os.path.dirname(path), exist_ok=True)
        q1_serialized = {str(k): v for k, v in self.q1.items()}
        q2_serialized = {str(k): v for k, v in self.q2.items()}
        with open(path, 'w') as f:
            json.dump({"q1": q1_serialized, "q2": q2_serialized}, f)

    def load(self, path):
        import json
        with open(path, 'r') as f:
            data = json.load(f)
            self.q1 = {eval(k): v for k, v in data["q1"].items()}
            self.q2 = {eval(k): v for k, v in data["q2"].items()}
