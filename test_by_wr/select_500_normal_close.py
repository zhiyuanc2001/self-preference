import sys
sys.path.append('/data/zhiyuan_data/SelfEvaluation')

import os
from utils import load_jsonl_data, add_jsonl_data
import json


def get_save_newfile(merge_path, evaluator_name, new_dir, all_ids):
    """golden label model"""
    file_name = os.path.basename(merge_path)

    save_path = os.path.join(new_dir, evaluator_name, file_name)
    if os.path.exists(save_path):
        raise

    print(f'Add file: {save_path}')

    # read jsonl file
    data = load_jsonl_data(merge_path)
    id2item = {item["id"]: item for item in data}
    new_data = []
    for id in all_ids:
        new_data.append(id2item[id])
    
    for new_item in new_data:
        add_jsonl_data(save_path=save_path, save_data=new_item)
        

####### Alpaca eval
with open('select_ids/alpaca_500_ids_v5.json', 'r') as rf:
    all_ids = json.load(rf)
original_dir = "model_preferences_fullset/alpaca_eval"
new_dir = "model_preferences_fullset/alpaca_eval_500_id_v5_wr"

# ###### Translation
# with open('select_ids/translation_500_ids_v2.json', 'r') as rf:
#     all_ids = json.load(rf)
# original_dir = "model_preferences_fullset/translation"
# new_dir = "model_preferences_fullset/translation_500_id_v2_wr"

# ####### Truthfulness
# with open('select_ids/truthfulness_500_ids_v1.json', 'r') as rf:
#     all_ids = json.load(rf)
# original_dir = "model_preferences_fullset/truthfulness"
# new_dir = "model_preferences_fullset/truthfulness_500_id_v1_wr"

for file_name in [
    # closed instruct-instruct
    "evaluator_qwen-plus/merge_qwen-plus_Llama-3.1-70B-Instruct.jsonl",
    "evaluator_glm-4-plus/merge_glm-4-plus_Llama-3.1-70B-Instruct.jsonl",
    "evaluator_claude-3.5-haiku/merge_claude-3.5-haiku_Llama-3.1-70B-Instruct.jsonl",
]:
    single_file_path = os.path.join(original_dir, file_name)
    evaluator_name = file_name.split('/')[0]
    get_save_newfile(single_file_path, evaluator_name, new_dir, all_ids)