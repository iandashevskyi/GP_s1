from agents.q_learning import QLearningAgent
from agents.lookahead import LookaheadAgent
from agents.rulebased_agent import RuleBasedAgent
from core.game import Game
from utils.io_utils import generate_table_name
from utils.logger import Logger

player1 = QLearningAgent("QLearningAgent", epsilon=1.0)
player2 = RuleBasedAgent("RuleBasedAgent")
players = [player1, player2]

filename = generate_table_name(player1.name, player2.name)
filepath = f"checkpoints/{filename}"
logpath = filepath.replace("checkpoints", "logs").replace(".json", ".csv")
logger = Logger(logpath, interval=10000)

EPSILON_MIN = 0.1
EPSILON_DECAY = 0.9

game = Game(players)

def evaluate(agent, opponent, episodes=1000):
    w = d = l = moves = 0

    for _ in range(episodes // 1):
        game1 = Game([agent, opponent])
        game1.play()
        moves += game1.moves_made
        if game1.winner == 0:
            w += 1
        elif game1.winner == 1:
            l += 1
        else:
            d += 1


    avg_moves = round(moves / episodes, 2)
    return w, d, l, avg_moves


for step in range(1, 100001):
    game.play()
    if game.winner == 0:
        player1.learn(1)
        player2.learn(-1)
    elif game.winner == 1:
        player1.learn(-1)
        player2.learn(1)
    else:
        player1.learn(0.5)
        player2.learn(0.5)

    logger.record(game.winner, player1.epsilon, game.moves_made)
    player1.epsilon = max(EPSILON_MIN, player1.epsilon * EPSILON_DECAY)
    game.reset()

    if step % 200 == 0:
        eval_agent = QLearningAgent("eval", epsilon=0.0)
        eval_agent.q_table = player1.q_table.copy()
        eval_w, eval_d, eval_l, eval_avg_moves = evaluate(eval_agent, player2)
        print(f"[Eval @ {step}] Wins: {eval_w} | Draws: {eval_d} | Losses: {eval_l} | Avg moves: {eval_avg_moves}" )

player1.save(filepath)
print(f"Q-table saved to {filepath}")
print(f"Training log saved to {logpath}")
