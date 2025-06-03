class Game:
    def __init__(self, players):
        self.players = players
        self.board = [" " for _ in range(9)]
        self.game_over = False
        self.winner = None
        self.moves_made = 0

    def play(self):
        self.moves_made = 0
        while not self.game_over:
            for idx, player in enumerate(self.players):
                move = player.make_move(self.board)
                self.board[move] = "X" if idx == 0 else "O"
                self.moves_made += 1
                self.check_for_win()
                if self.game_over:
                    return

    def check_for_win(self):
        win_conditions = [
            (0, 1, 2), (3, 4, 5), (6, 7, 8),
            (0, 3, 6), (1, 4, 7), (2, 5, 8),
            (0, 4, 8), (2, 4, 6)
        ]
        for i, j, k in win_conditions:
            if self.board[i] != " " and self.board[i] == self.board[j] == self.board[k]:
                self.winner = 0 if self.board[i] == "X" else 1
                self.game_over = True
                return
        if " " not in self.board:
            self.winner = 2  # draw
            self.game_over = True

    def reset(self):
        self.board = [" " for _ in range(9)]
        self.game_over = False
        self.winner = None
        self.moves_made = 0
