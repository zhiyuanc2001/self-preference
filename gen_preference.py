import os
import argparse
from tqdm import tqdm

from prompts.question_answering import(
    ALPACA_PREFERENCE_SYSTEM_PROMPT,
    ALPACA_PREFERENCE_USER_PROMPT,
    ALPACA_PREFERENCE_USER_PROMPT_ONLY,
    ALPACA_PREFERENCE_PROMPT_TWO_SHOT,
    ALPACA_PREFERENCE_USER_PROMPT_TWO_SHOT,
    ALPACA_PREFERENCE_USER_PROMPT_ONLY_TWO_SHOT
)
from prompts.translation import(
    TRANSLATION_PREFERENCE_USER_PROMPT,
    TRANSLATION_PREFERENCE_PROMPT_TWO_SHOT
)
from prompts.truthfulness import (
    TRUTHFULNESS_PREFERENCE_USER_PROMPT,
    TRUTHFULNESS_PREFERENCE_PROMPT_TWO_SHOT
)
from model_manager import (OpenAIManager, HFManager)
from utils import (load_jsonl_data, add_jsonl_data)


def parse_args():
    parser = argparse.ArgumentParser()
    # data args
    parser.add_argument("--data_dir", type=str, default=None, help="Base directory of test data.")
    parser.add_argument("--data_type", type=str, default="summarization", help="The type of dataset (e.g., summarization)")
    parser.add_argument("--model_name_1", type=str, default="gpt-4o-mini")
    parser.add_argument("--model_name_2", type=str, default="gpt-4")
    
    # evaluator args
    parser.add_argument("--model_type", type=str, default="gpt-4o-mini", help="Evaluator model type")
    parser.add_argument("--is_instruct", action='store_true', help="You should set to True if use GPT models.")
    parser.add_argument("--few_shot_instruct", action='store_true', help="Use few shot examples in `Instruct` model. Only useful when `is_instruct = True`")
    parser.add_argument("--use_infer_generate", action='store_true', help="Use `infer_generate()` instead of `prefer_generate()`.")
    parser.add_argument("--batch_size", type=int, default=1)
    return parser.parse_args()

def format_dataset_alpaca_instruct_model(data, system_prompt=None, meta_user_prompt: str = None):
    messages, ids, responses_1, responses_2 = [], [], [], []
    for data_item in data:
        user_prompt = meta_user_prompt.format(
            query=data_item["query"],
            response1=data_item["response1"],
            response2=data_item["response2"]
        )
        if system_prompt is None:
            msg = [
                {"role": "user", "content": user_prompt}
            ]
        else:
            msg = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
        messages.append(msg)
        ids.append(data_item["id"])
        responses_1.append(data_item["response1"])
        responses_2.append(data_item["response2"])
    return messages, ids, responses_1, responses_2

def format_dataset_alpaca_base_model(data, prompt):
    messages, ids, responses_1, responses_2 = [], [], [], []
    for data_item in data:
        msg = prompt.format(
            test_query=data_item["query"],
            test_response1=data_item["response1"],
            test_response2=data_item["response2"]
        )
        messages.append(msg)
        ids.append(data_item["id"])
        responses_1.append(data_item["response1"])
        responses_2.append(data_item["response2"])
    return messages, ids, responses_1, responses_2

def format_dataset_alpaca_instruct_model_few_shot(data, system_prompt, meta_user_prompt: str):
    messages, ids, responses_1, responses_2 = [], [], [], []
    for data_item in data:
        user_prompt = meta_user_prompt.format(
            test_query=data_item["query"],
            test_response1=data_item["response1"],
            test_response2=data_item["response2"]
        )
        if system_prompt is None:
            msg = [{"role": "user", "content": user_prompt}]
        else:
            msg = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
        messages.append(msg)
        ids.append(data_item["id"])
        responses_1.append(data_item["response1"])
        responses_2.append(data_item["response2"])
    return messages, ids, responses_1, responses_2

def format_dataset_translation_instruct_model(data, meta_user_prompt):
    messages, ids, responses_1, responses_2 = [], [], [], []
    for data_item in data:
        user_prompt = meta_user_prompt.format(
            german=data_item["german"],
            english1=data_item["response1"],
            english2=data_item["response2"]
        )
        msg = [
            {"role": "user", "content": user_prompt}
        ]
        messages.append(msg)
        ids.append(data_item["id"])
        responses_1.append(data_item["response1"])
        responses_2.append(data_item["response2"])
    return messages, ids, responses_1, responses_2

def format_dataset_translation_base_model(data, prompt):
    messages, ids, responses_1, responses_2 = [], [], [], []
    for data_item in data:
        msg = prompt.format(
            test_german=data_item["german"],
            test_english1=data_item["response1"],
            test_english2=data_item["response2"]
        )
        messages.append(msg)
        ids.append(data_item["id"])
        responses_1.append(data_item["response1"])
        responses_2.append(data_item["response2"])
    return messages, ids, responses_1, responses_2

def format_dataset_truthfulness_instruct_model(data, meta_user_prompt: str = None):
    messages, ids, responses_1, responses_2 = [], [], [], []
    for data_item in data:
        user_prompt = meta_user_prompt.format(
            query=data_item["query"],
            response1=data_item["response1"],
            response2=data_item["response2"]
        )
        msg = [
            {"role": "user", "content": user_prompt}
        ]
        messages.append(msg)
        ids.append(data_item["id"])
        responses_1.append(data_item["response1"])
        responses_2.append(data_item["response2"])
    return messages, ids, responses_1, responses_2

def format_dataset_truthfulness_base_model(data, prompt):
    messages, ids, responses_1, responses_2 = [], [], [], []
    for data_item in data:
        msg = prompt.format(
            test_query=data_item["query"],
            test_response1=data_item["response1"],
            test_response2=data_item["response2"]
        )
        messages.append(msg)
        ids.append(data_item["id"])
        responses_1.append(data_item["response1"])
        responses_2.append(data_item["response2"])
    return messages, ids, responses_1, responses_2


def gen_preference(compare_data: list[dict], model_type: str, is_instruct, 
                   few_shot_instruct, use_infer_generate, batch_size, data_type, save_path):
    if data_type == "alpaca_eval":
        print("Data type: alpaca_eval")
        if is_instruct:
            print('Instruct Model')
            if few_shot_instruct:
                print("Use few shot examples")
                if model_type.split('/')[-1] == "gemma-2-9b-it":
                    print('Customize user_prompt for model without `system_prompt`')
                    meta_user_prompt = ALPACA_PREFERENCE_USER_PROMPT_ONLY_TWO_SHOT
                    messages, ids, responses_1, responses_2 = format_dataset_alpaca_instruct_model_few_shot(
                        data=compare_data,
                        system_prompt=None,
                        meta_user_prompt=meta_user_prompt,
                    )
                else:
                    system_prompt = ALPACA_PREFERENCE_SYSTEM_PROMPT
                    meta_user_prompt = ALPACA_PREFERENCE_USER_PROMPT_TWO_SHOT
                    messages, ids, responses_1, responses_2 = format_dataset_alpaca_instruct_model_few_shot(
                        data=compare_data,
                        system_prompt=system_prompt,
                        meta_user_prompt=meta_user_prompt
                    )                
            else:
                print("Use zero shot example")
                if model_type.split('/')[-1] == "gemma-2-9b-it" or "-sft-UC" in model_type or "DeepSeek-R1-Distill-" in model_type:
                    print('Customize user_prompt for model without `system_prompt`')
                    meta_user_prompt = ALPACA_PREFERENCE_USER_PROMPT_ONLY
                    messages, ids, responses_1, responses_2 = format_dataset_alpaca_instruct_model(
                        data=compare_data,
                        system_prompt=None,
                        meta_user_prompt=meta_user_prompt,
                    )
                else:
                    system_prompt = ALPACA_PREFERENCE_SYSTEM_PROMPT
                    meta_user_prompt = ALPACA_PREFERENCE_USER_PROMPT
                    messages, ids, responses_1, responses_2 = format_dataset_alpaca_instruct_model(
                        data=compare_data,
                        system_prompt=system_prompt,
                        meta_user_prompt=meta_user_prompt,
                    )
        else:
            print('Base Model')
            prompt = ALPACA_PREFERENCE_PROMPT_TWO_SHOT
            messages, ids, responses_1, responses_2 = format_dataset_alpaca_base_model(
                data=compare_data,
                prompt=prompt
            )
    elif data_type == "translation":
        print('Data type: translation')
        if is_instruct:
            print("Instruction Model")
            if few_shot_instruct:
                raise NotImplementedError
            else:
                print("Use zero shot example")
                meta_user_prompt = TRANSLATION_PREFERENCE_USER_PROMPT
                messages, ids, responses_1, responses_2 = format_dataset_translation_instruct_model(
                    data=compare_data,
                    meta_user_prompt=meta_user_prompt,
                )
        else:
            print("Base Model")
            prompt = TRANSLATION_PREFERENCE_PROMPT_TWO_SHOT
            messages, ids, responses_1, responses_2 = format_dataset_translation_base_model(
                data=compare_data,
                prompt=prompt
            )
    elif data_type == "truthfulness":
        print("Data type: truthfulness")
        if is_instruct:
            print("Instruct Model")
            if few_shot_instruct:
                raise NotImplementedError
            else:
                print("Use zero shot example")
                meta_user_prompt = TRUTHFULNESS_PREFERENCE_USER_PROMPT
                messages, ids, responses_1, responses_2 = format_dataset_truthfulness_instruct_model(
                    data=compare_data,
                    meta_user_prompt=meta_user_prompt,
                )
        else:
            print("Base Model")
            prompt = TRUTHFULNESS_PREFERENCE_PROMPT_TWO_SHOT
            messages, ids, responses_1, responses_2 = format_dataset_truthfulness_base_model(
                data=compare_data,
                prompt=prompt
            )
    else:
        raise NotImplementedError
    
    max_tokens = 4
    temperature = 0
    logprobs = True
    top_logprobs = 2
    assert len(messages) == len(ids)
    
    if model_type.startswith("gpt") or model_type.startswith("gemini") or model_type.startswith("deepseek") or model_type == "qwen-plus" or model_type == "glm-4-plus":
        print(f"API model: {model_type}")
        assert is_instruct == True
        model_cls = OpenAIManager(model_type=model_type)
    else:
        print("Local model")
        model_cls = HFManager(
            model_name_or_path=model_type,
            is_instruct=is_instruct,
            bf16=True
        )

    num_batches = (len(messages) + batch_size - 1) // batch_size
    for i in tqdm(range(num_batches), desc="Generating preferences:"):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, len(messages))
        batch_messages = messages[start_idx:end_idx]

        if use_infer_generate:
            batch_responses = model_cls.infer_generate_parallel(
                messages=batch_messages,
                max_tokens=1,
                temperature=temperature,
                stop_words=None
            )

            for inner_idx in range(len(batch_responses)):
                outer_idx = start_idx + inner_idx
                inner_response = batch_responses[inner_idx]
                if inner_response is not None:
                    save_item = {
                        "response1": responses_1[outer_idx],
                        "response2": responses_2[outer_idx],
                        "model_preference": inner_response,
                        "id": ids[outer_idx]
                    }
                    add_jsonl_data(save_path=save_path, save_data=save_item)
        else:
            batch_responses = model_cls.prefer_generate(
                messages=batch_messages,
                max_tokens=max_tokens,
                temperature=temperature,
                logprobs=logprobs,
                top_logprobs=top_logprobs,
            )

            # save_data
            for inner_idx in range(len(batch_responses)):
                outer_idx = start_idx + inner_idx
                inner_response: list = batch_responses[inner_idx]
                if inner_response is not None:
                    save_item = {
                        "response1": responses_1[outer_idx],
                        "prob1": inner_response[0],
                        "response2": responses_2[outer_idx],
                        "prob2": inner_response[1],
                        "id": ids[outer_idx]
                    }
                    add_jsonl_data(save_path=save_path, save_data=save_item)


def merge_data_alpaca(data1, data2):
    list2_dict = {item["id"]: item["model_response"] for item in data2}
    merged_list = []
    for item in data1:
        item_id = item["id"]
        if item_id in list2_dict:
            merged_item = {
                "response1": item["model_response"],
                "response2": list2_dict[item_id],
                "golden_response": item["golden_response"],
                "query": item["query"],
                "id": item_id
            }
            merged_list.append(merged_item)
            
    return merged_list


def merge_data_translaiton(data1, data2):
    list2_dict = {item["id"]: item["model_response"] for item in data2}
    merged_list = []
    for item in data1:
        item_id = item["id"]
        if item_id in list2_dict:
            merged_item = {
                "response1": item["model_response"],
                "response2": list2_dict[item_id],
                "golden_response": item["golden_response"],
                "german": item["german"],
                "id": item_id
            }
            merged_list.append(merged_item)
            
    return merged_list

def merge_data_truthfulness(data1, data2):
    list2_dict = {item["id"]: item["model_response"] for item in data2}
    merged_list = []
    for item in data1:
        item_id = item["id"]
        if item_id in list2_dict:
            merged_item = {
                "response1": item["model_response"],
                "response2": list2_dict[item_id],
                "golden_response": item["golden_response"],
                "query": item["query"],
                "id": item_id
            }
            merged_list.append(merged_item)
            
    return merged_list


if __name__ == "__main__":
    args = parse_args()
    data_path_1 = os.path.join(args.data_dir, args.model_name_1 + ".jsonl")
    data_path_2 = os.path.join(args.data_dir, args.model_name_2 + ".jsonl")
    
    # load data
    data1 = load_jsonl_data(data_path_1)
    data2 = load_jsonl_data(data_path_2)
    if args.data_type == "alpaca_eval":
        merge_data_func = merge_data_alpaca
    elif args.data_type == "translation":
        merge_data_func = merge_data_translaiton
    elif args.data_type == "truthfulness":
        merge_data_func = merge_data_truthfulness

    data_merge = merge_data_func(data1, data2)
    print(f"Total {len(data_merge)} items to be compared.")
    
    evaluator_name = args.model_type.split('/')[-1]
    if evaluator_name == "gpt-4o-mini":
        evaluator_name = "gpt-4o-mini_choose"

    if args.few_shot_instruct:
        save_path = f"model_preferences_fullset/{args.data_type}/evaluator_{evaluator_name}_fewshot/{args.model_name_1}_{args.model_name_2}.jsonl"
    else:
        save_path = f"model_preferences_fullset/{args.data_type}/evaluator_{evaluator_name}/{args.model_name_1}_{args.model_name_2}.jsonl"
    
    if os.path.exists(save_path):
        raise ValueError("Path exists!")
    
    gen_preference(
        compare_data=data_merge,
        model_type=args.model_type,
        is_instruct=args.is_instruct,
        few_shot_instruct=args.few_shot_instruct,
        use_infer_generate=args.use_infer_generate,
        batch_size=args.batch_size,
        data_type=args.data_type,
        save_path=save_path
    )