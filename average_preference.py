"""Since the probabilities of the two options may not add up to one 1, here we normalize the two probabilities"""

import os
import argparse
from utils import load_jsonl_data, add_jsonl_data


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--evaluation_type", type=str, default="summarization")
    parser.add_argument("--evaluator_name", type=str, default="gpt-4o-mini")
    parser.add_argument("--model_name_1", type=str, default="gpt-4o-mini")
    parser.add_argument("--model_name_2", type=str, default="Llama-3.1-8B-Instruct")
    parser.add_argument("--few_shot_instruct", action='store_true', help="Use few shot examples in `Instruct` model. Only useful when `is_instruct = True`")

    return parser.parse_args()


def get_ids(data):
    ids = []
    for item in data:
        ids.append(item['id'])     
    return ids

def transform_data(data: list[dict]):
    trans_data = {}
    for sub_data in data:
        sub_id = sub_data["id"]
        trans_data[sub_id] = sub_data
    
    return trans_data


if __name__ == "__main__":
    args = parse_args()
    evaluation_type = args.evaluation_type
    evaluator_name = args.evaluator_name

    model_name_1 = args.model_name_1
    model_name_2 = args.model_name_2

    if args.few_shot_instruct:
        evaluate_dir = f"model_preferences_fullset/{evaluation_type}/evaluator_{evaluator_name}_fewshot"
    else:
        evaluate_dir = f"model_preferences_fullset/{evaluation_type}/evaluator_{evaluator_name}"

    output_file_name = f"average_{model_name_1}_{model_name_2}.jsonl"

    data_1 = load_jsonl_data(os.path.join(evaluate_dir, f"{model_name_1}_{model_name_2}.jsonl"))
    data_2 = load_jsonl_data(os.path.join(evaluate_dir, f"{model_name_2}_{model_name_1}.jsonl"))

    # get ids
    data_1_ids = get_ids(data_1)
    data_2_ids = get_ids(data_2)
    intersection_ids = [subid for subid in data_1_ids if subid in data_2_ids]
    print(f"intersection id number : {len(intersection_ids)}")

    trans_data_1 = transform_data(data_1)
    trans_data_2 = transform_data(data_2)
    
    save_path = os.path.join(evaluate_dir, output_file_name)
    if os.path.exists(save_path):
        raise
    print(f'Saving path to: {save_path}')

    for sub_id in intersection_ids:
        save_item = {}
        save_item["id"] = sub_id
        sub_data_1 = trans_data_1[sub_id]
        sub_data_2 = trans_data_2[sub_id]
        
        # normalize
        sub_data_1_prob1 = sub_data_1["prob1"] / (sub_data_1['prob1'] + sub_data_1['prob2'])
        sub_data_1_prob2 = sub_data_1['prob2'] / (sub_data_1['prob1'] + sub_data_1['prob2'])
        
        sub_data_2_prob1 = sub_data_2['prob1'] / (sub_data_2['prob1'] + sub_data_2['prob2'])
        sub_data_2_prob2 = sub_data_2['prob2'] / (sub_data_2['prob1'] + sub_data_2['prob2'])
        
        
        # prob of response"1"
        assert sub_data_1["response1"] == sub_data_2["response2"]
        save_item[f"{model_name_1}_response"] = sub_data_1["response1"]
        save_item[f"{model_name_1}_prob"] = (sub_data_1_prob1 + sub_data_2_prob2) / 2
        
        assert sub_data_1["response2"] == sub_data_2["response1"]
        save_item[f"{model_name_2}_response"] = sub_data_1["response2"]
        save_item[f"{model_name_2}_prob"] = (sub_data_1_prob2 + sub_data_2_prob1) / 2
        
        add_jsonl_data(save_path=save_path, save_data=save_item)
