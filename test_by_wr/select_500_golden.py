import sys
sys.path.append('your path')

import os
import json
from utils import load_jsonl_data, add_jsonl_data


def get_save_newfile(merge_path, evaluator_name, new_dir, all_ids):
    """golden label model"""
    file_name = os.path.basename(merge_path)
    if evaluator_name == "evaluator_gpt-4o-mini_choose":
        evaluator_name = "evaluator_gpt-4o-mini"
    save_path = os.path.join(new_dir, evaluator_name + '_golden', file_name)
    if os.path.exists(save_path):
        print(f"Warning:: {save_path} exists, skip it")
        return

    print(f'Add file: {save_path}')

    # read jsonl file
    data = load_jsonl_data(merge_path)
    id2item = {item["id"]: item for item in data}
    new_data = []
    for id in all_ids:
        new_data.append(id2item[id])
    
    for new_item in new_data:
        add_jsonl_data(save_path=save_path, save_data=new_item)


###### alpaca eval
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



golden_evaluators = ["evaluator_gpt-4o-mini_choose", "evaluator_gemini-flash-1.5", "evaluator_deepseek-v3"]

pair_lists = [
    # # ## base-instruct
    # 'merge_Llama-3.1-8B_Llama-3.1-8B-Instruct.jsonl',
    # 'merge_Llama-3.1-70B_Llama-3.1-70B-Instruct.jsonl',
    # 'merge_Qwen2.5-7B_Qwen2.5-7B-Instruct.jsonl',
    # 'merge_Qwen2.5-72B_Qwen2.5-72B-Instruct.jsonl',
    # 'merge_gemma-2-9b_gemma-2-9b-it.jsonl',
    # # ## base-base
    # 'merge_Llama-3.1-8B_Qwen2.5-7B.jsonl',
    # 'merge_Llama-3.1-70B_Qwen2.5-72B.jsonl',
    # 'merge_Llama-3.1-8B_gemma-2-9b.jsonl',
    # # ## instruct-instruct
    'merge_claude-3.5-haiku_Llama-3.1-70B-Instruct.jsonl',
    'merge_glm-4-plus_Llama-3.1-70B-Instruct.jsonl',
    'merge_qwen-plus_Llama-3.1-70B-Instruct.jsonl',
    
    # 'merge_Llama-3.1-8B-Instruct_Qwen2.5-7B-Instruct.jsonl',
    # 'merge_Llama-3.1-70B-Instruct_Qwen2.5-72B-Instruct.jsonl',
    # 'merge_Llama-3.1-8B-Instruct_gemma-2-9b-it.jsonl',
    # # ## small-big
    # 'merge_Llama-3.1-70B_Llama-3.1-8B.jsonl',
    # 'merge_Qwen2.5-72B_Qwen2.5-7B.jsonl',
    # 'merge_Llama-3.1-70B-Instruct_Llama-3.1-8B-Instruct.jsonl',
    # 'merge_Qwen2.5-72B-Instruct_Qwen2.5-7B-Instruct.jsonl',
    
    # # Alpaca only
    # ## mytrain-mytrain
    # 'merge_Llama-3.1-8B-sft-UCfull_Qwen2.5-7B-sft-UCfull.jsonl',
    # 'merge_Llama-3.1-8B-Instruct_Llama-3.1-Tulu-3-8B.jsonl',
    
    ## scaling
    # 'merge_Llama-3.1-70B-Instruct_Qwen2.5-0.5B-Instruct.jsonl',
    # 'merge_Llama-3.1-70B-Instruct_Qwen2.5-1.5B-Instruct.jsonl',
    # 'merge_Llama-3.1-70B-Instruct_Qwen2.5-3B-Instruct.jsonl',
    # 'merge_Llama-3.1-70B-Instruct_Qwen2.5-7B-Instruct.jsonl',
    # 'merge_Llama-3.1-70B-Instruct_Qwen2.5-14B-Instruct.jsonl',
    # 'merge_Llama-3.1-70B-Instruct_Qwen2.5-32B-Instruct.jsonl',
]

for evaluator in golden_evaluators:
    for pair in pair_lists:
        single_file_path = os.path.join(original_dir, evaluator, pair)
        get_save_newfile(single_file_path, evaluator, new_dir, all_ids)