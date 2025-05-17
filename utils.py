import json
import os


def load_json_data(load_path):
    with open(load_path, "r", encoding="utf-8") as rf:
        data = json.load(rf)
    return data

def save_json_data(save_path, save_data):
    save_dir = os.path.dirname(save_path)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    with open(save_path, "w", encoding="utf-8") as wf:
        json.dump(save_data, wf, indent=2, ensure_ascii=False)


def load_jsonl_data(load_path: str):
    data = []
    with open(load_path, "r", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line))
    return data


def add_jsonl_data(save_path: str, save_data: dict):
    save_dir = os.path.dirname(save_path)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    with open(save_path, 'a', encoding='utf-8') as wf:
        wf.write(json.dumps(save_data) + '\n')