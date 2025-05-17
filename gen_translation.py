import os
import argparse
from tqdm import tqdm

from datasets import load_from_disk
from prompts.translation import(
    TRANSLATION_USER_PROMPT,
    TRANSLATION_PROMPT_TWO_SHOT
)
from model_manager import (OpenAIManager, HFManager)
from utils import add_jsonl_data


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="datasets/cnn_dailymail")
    parser.add_argument("--model_type", type=str, default="gpt-4o-mini")
    parser.add_argument("--is_instruct", action='store_true', help="You should set to True if use GPT models.")
    parser.add_argument("--batch_size", type=int, default=1)
    return parser.parse_args()


def format_dataset_instruct_model(data, meta_user_prompt: str):
    messages, ids, golden_responses, german = [], [], [], []
    for item in data:
        user_prompt = meta_user_prompt.format(german=item["de"])
        msg = [
            {"role": "user", "content": user_prompt}
        ]
        messages.append(msg)
        ids.append(item["id"])
        golden_responses.append(item["en"])
        german.append(item["de"])
    return messages, ids, golden_responses, german


def format_dataset_base_model(data, prompt: str):
    messages, ids, golden_responses, german = [], [], [], []
    for item in data:
        msg = prompt.format(test_german=item["de"])
        messages.append(msg)
        ids.append(item["id"])
        golden_responses.append(item["en"])
        german.append(item["de"])
    return messages, ids, golden_responses, german


def generate_translation(data_dir, model_type, is_instruct, batch_size):
    """
        generate english tranlation for translation task.
        data_dir (`str`): Data path.
        model_type (`str`): Models for generating responses.
    """
    data_type = os.path.basename(data_dir)
    model_name = model_type.split('/')[-1]
    if model_name == "llama3_1-8B-hf-Chat":
        model_name = "Llama-3.1-8B-Instruct"
    elif model_name == "llama3_1-8B-hf":
        model_name = "Llama-3.1-8B"

    save_path = f"model_responses_fullset/translation/{data_type}/{model_name}.jsonl"
    # load data
    test_data = load_from_disk(data_dir)
    print(f'Test data numbers: {len(test_data)}')

    if data_type == "wmt_de-en":
        if is_instruct == True:
            print("Instruct Model")
            meta_user_prompt = TRANSLATION_USER_PROMPT
            messages, ids, golden_responses, german = format_dataset_instruct_model(test_data, meta_user_prompt)
            stop_words = None
        else:
            print("Base Model")
            prompt = TRANSLATION_PROMPT_TWO_SHOT
            messages, ids, golden_responses, german = format_dataset_base_model(test_data, prompt)
            stop_words = ["# German"]
            assert len(stop_words) == 1
        max_tokens = 128
        temperature = 0.0
        print(f"Max tokens: {max_tokens} || Temperature: {temperature}")
        assert len(messages) == len(ids)
    else:
        raise ValueError(f'The data_type: `{data_type}` is not supported.')
    
    if model_type.startswith('gpt'):
        print(f'API Model: {model_name}')
        assert is_instruct == True
        model_cls = OpenAIManager(model_type=model_type)
    else:
        print(f'Local Model: {model_name}')
        model_cls = HFManager(model_name_or_path=model_type,
                              is_instruct=is_instruct,
                              bf16=True)
    

    num_batches = (len(messages) + batch_size - 1) // batch_size        
    for i in tqdm(range(num_batches), desc="Generating model responses:"):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, len(messages))
        batch_messages = messages[start_idx:end_idx]
        # get response
        batch_responses = model_cls.infer_generate(
            messages=batch_messages,
            max_tokens=max_tokens,
            temperature=temperature,
            stop_words=stop_words
        )
        assert len(batch_responses) == len(batch_messages)
        
        # save_data
        for inner_idx in range(len(batch_responses)):
            outer_idx = start_idx + inner_idx
            inner_response = batch_responses[inner_idx]
            if inner_response is not None:
                save_item = {
                    "model_response": inner_response,
                    "golden_response": golden_responses[outer_idx],
                    "german": german[outer_idx],
                    "id": ids[outer_idx]
                }
                add_jsonl_data(save_path=save_path, save_data=save_item)


if __name__ == "__main__":
    args = parse_args()
    
    generate_translation(
        data_dir=args.data_dir,
        model_type=args.model_type,
        is_instruct=args.is_instruct,
        batch_size=args.batch_size
    )
