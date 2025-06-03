from agents.sarsa import SarsaAgent
from agents.random_agent import RandomAgent
from agents.q_learning import QLearningAgent
from agents.lookahead import LookaheadAgent
from agents.rulebased_agent import RuleBasedAgent
from agents.double_q_learning import DoubleQLearningAgent
from core.game import Game
from datetime import datetime
import os


def load_agent(agent_type, path):
    if agent_type == "q":
        agent = QLearningAgent("QLearningAgent", epsilon=0)
    elif agent_type == "sarsa":
        agent = SarsaAgent("SarsaAgent", epsilon=0)
    elif agent_type == "double":
        agent = DoubleQLearningAgent("DoubleQ", epsilon=0)
    else:
        raise ValueError("Unknown RL agent type")

    agent.load(path)
    return agent


def create_baseline(name):
    if name == "random":
        return RandomAgent("RandomAgent")
    elif name == "rule":
        return RuleBasedAgent("RuleBasedAgent")
    elif name == "lookahead":
        return LookaheadAgent("LookaheadAgent")
    else:
        raise ValueError("Unknown baseline agent type")

def evaluate_agent(agent, opponent, episodes=10000):
    wins = draws = losses = total_moves = 0

    for _ in range(episodes // 2):
        # agent first
        game1 = Game([agent, opponent])
        game1.play()
        total_moves += game1.moves_made
        if game1.winner == 0:
            wins += 1
        elif game1.winner == 1:
            losses += 1
        else:
            draws += 1

        # agent second
        game2 = Game([opponent, agent])
        game2.play()
        total_moves += game2.moves_made
        if game2.winner == 1:
            wins += 1
        elif game2.winner == 0:
            losses += 1
        else:
            draws += 1

    avg_moves = round(total_moves / episodes, 2)
    return wins, losses, draws, avg_moves


def main():
    t1 = input("Enter agent 1 type (q/sarsa/double): ").strip().lower()
    p1 = input("Enter agent 1 checkpoint path: ").strip()
    t2 = input("Enter agent 2 type (q/sarsa/double/random/rule/lookahead): ").strip().lower()

    player1 = load_agent(t1, p1)
    if t2 in ["q", "sarsa", "double"]:
        p2 = input("Enter agent 2 checkpoint path: ").strip()
        player2 = load_agent(t2, p2)
    else:
        player2 = create_baseline(t2)

    wins, losses, draws, avg_moves = evaluate_agent(player1, player2)

    result = f"Evaluation result: wins={wins}, losses={losses}, draws={draws}, avg_moves={avg_moves}"
    print(result)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    test_name = f"{player1.name}_vs_{player2.name}_{timestamp}.txt"
    os.makedirs("logs", exist_ok=True)
    with open(f"logs/{test_name}", "w") as f:
        f.write(result + "\n")



if __name__ == "__main__":
    main()
