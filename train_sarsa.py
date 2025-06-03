from agents.rulebased_agent import RuleBasedAgent
from agents.sarsa import SarsaAgent
from agents.lookahead import LookaheadAgent
from core.game import Game
from utils.io_utils import generate_table_name
from utils.logger import Logger

player1 = SarsaAgent("SarsaAgent", epsilon=1.0)
player2 = LookaheadAgent("LookaheadAgent")
players = [player1, player2]

filename = generate_table_name(player1.name, player2.name)
filepath = f"checkpoints/{filename}"
logpath = filepath.replace("checkpoints", "logs").replace(".json", ".csv")
logger = Logger(logpath, interval=10000)

EPSILON_MIN = 0.1
EPSILON_DECAY = 0.99

game = Game(players)

def evaluate(agent, opponent, episodes=1000):
    eval_game = Game([agent, opponent])
    w = d = l = moves = 0
    for _ in range(episodes):
        eval_game.play()
        moves += eval_game.moves_made
        if eval_game.winner == 0:
            w += 1
        elif eval_game.winner == 1:
            l += 1
        else:
            d += 1
        eval_game.reset()
    avg_moves = round(moves / episodes, 2)
    return w, d, l, avg_moves

for step in range(1, 100001):
    game.reset()
    player1.reset()
    prev_state = None
    prev_action = None

    while not game.game_over:
        state = tuple(game.board)
        action = player1.make_move(game.board.copy())
        game.board[action] = "X"
        game.check_for_win()

        if game.game_over:
            reward = 1 if game.winner == 0 else 0
            if prev_state is not None and prev_action is not None:
                player1.learn(reward, None, None)
            break

        move2 = player2.make_move(game.board)
        game.board[move2] = "O"
        game.check_for_win()

        if game.game_over:
            reward = -1 if game.winner == 1 else 0
            if prev_state is not None and prev_action is not None:
                player1.learn(reward, None, None)
            break

        next_state = tuple(game.board)
        next_action = player1.make_move(game.board.copy())

        if prev_state is not None and prev_action is not None:
            player1.learn(0, next_state, next_action)

        prev_state = next_state
        prev_action = next_action

    logger.record(game.winner, player1.epsilon, game.moves_made)
    player1.epsilon = max(EPSILON_MIN, player1.epsilon * EPSILON_DECAY)

    if step % 200 == 0:
        eval_agent = player1.clone_eval_agent()
        eval_w, eval_d, eval_l, eval_avg_moves = evaluate(eval_agent, player2)
        print(f"[Eval @ {step}] Wins: {eval_w} | Draws: {eval_d} | Losses: {eval_l} | Avg moves: {eval_avg_moves}")

player1.save(filepath)
print(f"Q-table saved to {filepath}")
print(f"Training log saved to {logpath}")
