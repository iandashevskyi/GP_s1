class BaseAgent:
    def make_move(self, board):
        raise NotImplementedError

    def learn(self, reward):
        pass
