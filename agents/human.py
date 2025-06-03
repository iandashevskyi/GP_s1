from agents.q_learning import QLearningAgent
from agents.sarsa import SarsaAgent
from core.game import Game

class HumanAgent:
    def __init__(self, name="You"):
        self.name = name

    def make_move(self, board):
        valid_moves = [i for i, val in enumerate(board) if val == " "]
        print("\nCurrent board:")
        self.print_board(board)
        print(f"Available moves: {valid_moves}")
        move = -1
        while move not in valid_moves:
            try:
                move = int(input("Make a move (0-8): "))
            except ValueError:
                continue
        return move

    def print_board(self, board):
        for i in range(3):
            print("|".join(board[i*3:(i+1)*3]))
            if i < 2:
                print("-----")

agent_type = input("Choose agent (q/sarsa): ").strip().lower()
path = input("Path to saved Q-table: ").strip()

iq_agent = QLearningAgent("AI") if agent_type == "q" else SarsaAgent("AI")
iq_agent.load(path)
human = HumanAgent()

first = input("Do you want to go first? (y/n): ").strip().lower() == 'y'
players = [human, iq_agent] if first else [iq_agent, human]

print("starting game You are 'X' play for white, 'O' for blacks.")
game = Game(players)
game.play()

if game.winner == 2:
    print("draw")
else:
    print(f"Winner: {players[game.winner].name}")
