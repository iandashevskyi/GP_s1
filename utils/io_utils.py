import json
import os
from datetime import datetime

def save_q_table(q_table, path):
    with open(path, 'w') as f:
        serializable_table = {str(k): v for k, v in q_table.items()}
        json.dump(serializable_table, f)

def load_q_table(path):
    if not os.path.exists(path):
        return {}
    with open(path, 'r') as f:
        raw = json.load(f)
        q_table = {eval(k): v for k, v in raw.items()}
    return q_table

def generate_table_name(agent1_type: str, agent2_type: str) -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{agent1_type}_vs_{agent2_type}_{timestamp}.json"