import csv
import os

class Logger:
    def __init__(self, path, interval=10000):
        self.path = path
        self.interval = interval
        self.counter = 0
        self.stats = {"wins": 0, "draws": 0, "losses": 0, "total_moves": 0}

        header = ["step", "wins", "draws", "losses", "avg_moves", "epsilon"]
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(self.path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(header)

    def record(self, result, epsilon, moves):
        self.counter += 1
        self.stats["total_moves"] += moves
        if result == 0:
            self.stats["wins"] += 1
        elif result == 1:
            self.stats["losses"] += 1
        else:
            self.stats["draws"] += 1

        if self.counter % self.interval == 0:
            self._flush(epsilon)

    def _flush(self, epsilon):
        avg_moves = round(self.stats["total_moves"] / self.interval, 2)
        with open(self.path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                self.counter,
                self.stats["wins"],
                self.stats["draws"],
                self.stats["losses"],
                avg_moves,
                round(epsilon, 4)
            ])
        self.stats = {"wins": 0, "draws": 0, "losses": 0, "total_moves": 0}
