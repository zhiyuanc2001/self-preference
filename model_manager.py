from openai import AzureOpenAI, OpenAI
from zhipuai import ZhipuAI
from concurrent.futures import ThreadPoolExecutor, as_completed
from tenacity import (
    retry,
    stop_after_attempt, # type: ignore
    wait_random_exponential, # type: ignore
)
import torch
import torch.nn.functional as F
from math import exp
from transformers import(
    AutoTokenizer,
    AutoModelForCausalLM,
    StoppingCriteria,
    StoppingCriteriaList,
)


class EndOfFunctionCriteria(StoppingCriteria):
    """
        Custom `StoppingCriteria` which checks if all generated functions in the batch are completed.
        Copied from github repo of
        ```
            @inproceedings{
                Lin2024ReAlign,
                title={The Unlocking Spell on Base LLMs: Rethinking Alignment via In-Context Learning},
                author={Bill Yuchen Lin and Abhilasha Ravichander and Ximing Lu and Nouha Dziri and Melanie Sclar and Khyathi Chandu and Chandra Bhagavatula and Yejin Choi},
                booktitle={International Conference on Learning Representations},
                year={2024},
                url={https://arxiv.org/abs/2312.01552}
            }
        ```
    """

    def __init__(self, start_length, eof_strings, tokenizer):
        self.start_length = start_length
        self.eof_strings = eof_strings
        self.tokenizer = tokenizer

    def __call__(self, input_ids, scores, **kwargs):
        """Returns true if all generated sequences contain any of the end-of-function strings."""
        decoded_generations = self.tokenizer.batch_decode(
            input_ids[:, self.start_length :]
        )
        done = []
        for decoded_generation in decoded_generations:
            done.append(
                any(
                    [
                        stop_string in decoded_generation
                        for stop_string in self.eof_strings
                    ]
                )
            )
        return all(done)


class OpenAIManager():
    def __init__(self, model_type: str) -> None:
        if model_type == "gpt-4o-mini":
            model_type = "openai/gpt-4o-mini-2024-07-18"
        elif model_type == "gpt-4o":
            model_type = "openai/gpt-4o-2024-08-06"
        elif model_type == "gemini-flash-1.5":
            model_type = "google/gemini-flash-1.5"
        elif model_type == "deepseek-v3":
            model_type = "deepseek-chat"
        elif model_type == "qwen-plus":
            model_type = "qwen/qwen-plus"
        elif model_type == "glm-4-plus":
            model_type = "glm-4-plus"
        else:
            raise NotImplementedError

        self.model_type = model_type
        if model_type == "deepseek-chat":
            self.openai_client = OpenAI(
                base_url="https://api.deepseek.com",
                api_key="Your deepseek key"
            )
        elif model_type == "glm-4-plus":
            self.openai_client = ZhipuAI(api_key="Your zhipu key")
        else:
            self.openai_client = OpenAI(
                base_url="https://openrouter.ai/api/v1",
                api_key="Your openrouter key",
            )

    # @retry(wait=wait_random_exponential(min=1, max=3), stop=stop_after_attempt(6))
    def infer_generate(self, messages: list[list], max_tokens=256, temperature=0.7, stop_words=None):
        outputs = []
        for msg in messages:
            try:
                rsp = self.openai_client.chat.completions.create(
                    model=self.model_type,
                    messages=msg,
                    max_tokens=max_tokens,
                    temperature=temperature,
                )
                output = rsp.choices[0].message.content
            except Exception as e:
                output = None
                print(f"API server generate an error message: {e}")
            outputs.append(output)
        return outputs


    @retry(wait=wait_random_exponential(min=1, max=5), stop=stop_after_attempt(2))
    def single_infer(self, msg, max_tokens=256, temperature=0.7):
        """Handles a single inference request."""
        try:
            rsp = self.openai_client.chat.completions.create(
                model=self.model_type,
                messages=msg,
                max_tokens=max_tokens,
                temperature=temperature,
            )
            return rsp.choices[0].message.content
        except Exception as e:
            print(f"API server generate an error message: {e}")
            return None
    
    def infer_generate_parallel(self, messages: list[list], max_tokens=256, temperature=0.7, stop_words=None):
        num_workers = len(messages)
        outputs = [None] * len(messages)  # Preallocate list to maintain order
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            future_to_index = {
                executor.submit(self.single_infer, msg, max_tokens, temperature): i
                for i, msg in enumerate(messages)
            }
            for future in as_completed(future_to_index):
                idx = future_to_index[future]  # Get the original index of the message
                try:
                    outputs[idx] = future.result()
                except Exception as e:
                    print(f"An error occurred: {e}")
                    outputs[idx] = None  # Handle failed tasks gracefully
        return outputs


    def prefer_generate_series(
        self,
        messages: list[list],
        max_tokens=256,
        temperature=0.0,
        logprobs=True,
        top_logprobs=2
    ):
        if top_logprobs != 2:
            raise NotImplementedError("Assign log probs to tokens more than 2 is not implemented.")
        
        outputs = []
        for msg in messages:
            try:
                rsp = self.openai_client.chat.completions.create(
                    model=self.model_type,
                    messages=msg,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    logprobs=logprobs,
                    top_logprobs=top_logprobs
                )
                output = rsp.choices[0].logprobs.content[0].top_logprobs
                
                ## post process
                output_prob = [0] * top_logprobs
                # If the second token is not A or B, the probability that it is A or B is essentially 0
                for top_token in output:
                    if top_token.token == "A":
                        output_prob[0] = exp(top_token.logprob)
                    elif top_token.token == "B":
                        output_prob[1] = exp(top_token.logprob)    
            except Exception as e:
                output_prob = None
                print(f"API server generate an error: {e}")

            outputs.append(output_prob)    
        return outputs


    def process_message(self, msg, max_tokens, temperature, logprobs, top_logprobs):
        """Process a single message with the API call and post-process."""
        try:
            rsp = self.openai_client.chat.completions.create(
                model=self.model_type,
                messages=msg,
                max_tokens=max_tokens,
                temperature=temperature,
                logprobs=logprobs,
                top_logprobs=top_logprobs
            )
            output = rsp.choices[0].logprobs.content[0].top_logprobs
            
            # Post process
            output_prob = [0] * top_logprobs
            for top_token in output:
                if top_token.token == "A":
                    output_prob[0] = exp(top_token.logprob)
                elif top_token.token == "B":
                    output_prob[1] = exp(top_token.logprob)
        except Exception as e:
            output_prob = None
            print(f"API server generated an error: {e}")
        return output_prob

    def prefer_generate(
        self,
        messages: list[list],
        max_tokens=256,
        temperature=0.0,
        logprobs=True,
        top_logprobs=2
    ):
        outputs = [None] * len(messages)  # Preallocate to maintain order
        with ThreadPoolExecutor(max_workers=len(messages)) as executor:
            future_to_index = {
                executor.submit(self.process_message, msg, max_tokens, temperature, logprobs, top_logprobs): i
                for i, msg in enumerate(messages)
            }
            for future in as_completed(future_to_index):
                idx = future_to_index[future]  # Get the original index
                try:
                    outputs[idx] = future.result()
                except Exception as e:
                    print(f"An error occurred while processing message {idx}: {e}")
                    outputs[idx] = None  # Fallback to None for failed tasks
        return outputs



class HFManager():
    def __init__(self, model_name_or_path: str, is_instruct: bool = False, bf16: bool = True) -> None:
        print(f'Load tokenzier and model from {model_name_or_path}')
        self.model_name = model_name_or_path.split('/')[-1]

        self.is_instruct = is_instruct
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path,
            trust_remote_code=True,
            padding_side="left"
        )
        if self.tokenizer is not None: 
            if self.tokenizer.pad_token_id is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            if "gemma-2" in model_name_or_path.lower():
                self.tokenizer.padding_side = "right"
        
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            torch_dtype=torch.bfloat16 if bf16 else torch.float16,
            trust_remote_code=True,
            device_map="auto"
        )
        self.model.eval()
    
    def infer_generate(
        self,
        messages,
        max_tokens=256,
        temperature=0.7,
        stop_words=None
    ):
        if self.is_instruct:
            device = self.model.device
            promtps = []
            for msg in messages:
                template_msg = self.tokenizer.apply_chat_template(
                    msg,
                    tokenize=False,
                    add_generation_prompt=True
                )
                promtps.append(template_msg)
            
            if len(promtps) > 1:
                padding = True
            else:
                padding = False
            
            inputs = self.tokenizer(promtps, return_tensors="pt",
                                    padding=padding, truncation=True, add_special_tokens=False)
                                    
            prompt_length = inputs['input_ids'].shape[1]
            gen_do_sample = True if temperature > 0 else False

            outputs = self.model.generate(
                input_ids=inputs["input_ids"].to(device),
                attention_mask=inputs["attention_mask"].to(device),
                pad_token_id=self.tokenizer.pad_token_id,
                tokenizer=self.tokenizer,
                do_sample=gen_do_sample,
                temperature=temperature if gen_do_sample else None,
                max_new_tokens=max_tokens,
                top_p=1.0,
                top_k=None,
            )
            outputs = [self.tokenizer.decode(op[prompt_length:], skip_special_tokens=True).strip() for op in outputs]
        else:
            device = self.model.device
            if len(messages) > 1:
                padding = True
            else:
                padding = False
            
            inputs = self.tokenizer(messages, return_tensors="pt",
                                    add_special_tokens=True, padding=padding)
            
            prompt_length = inputs["input_ids"].shape[1]
            stopping_criteria = StoppingCriteriaList([EndOfFunctionCriteria(start_length=prompt_length, 
                                                                            eof_strings=stop_words, 
                                                                            tokenizer=self.tokenizer)])
            
            gen_do_sample = True if temperature > 0 else False
            model_outputs = self.model.generate(
                input_ids=inputs['input_ids'].to(device),
                attention_mask=inputs["attention_mask"].to(device),
                pad_token_id=self.tokenizer.pad_token_id,
                do_sample=gen_do_sample,
                temperature=temperature if gen_do_sample else None,
                max_new_tokens=max_tokens,
                top_p=1.0,
                stopping_criteria=stopping_criteria,
            )
            model_outputs = [self.tokenizer.decode(op[prompt_length:], skip_special_tokens=True) for op in model_outputs]
            
            # post process stop_criteria
            outputs = []
            for op in model_outputs:
                # may not suitable for stop_words more than 2
                outputs.append(op.rstrip(stop_words[0]).strip())
            
        return outputs


    def prefer_generate(
        self,
        messages: list,
        max_tokens=4,
        temperature=0,
        logprobs=True,
        top_logprobs=2,
    ):
        assert len(messages) == 1
        device = self.model.device
    
        if self.is_instruct:
            promtps = []
            for msg in messages:
                template_msg = self.tokenizer.apply_chat_template(
                    msg,
                    tokenize=False,
                    add_generation_prompt=True
                )
                promtps.append(template_msg)
            if len(promtps) > 1:
                padding = True
            else:
                padding = False
            inputs = self.tokenizer(promtps, return_tensors="pt",
                                    padding=padding, truncation=True, add_special_tokens=False)
        else:
            if len(messages) > 1:
                padding = True
            else:
                padding = False
            inputs = self.tokenizer(messages, return_tensors="pt",
                                    padding=padding, add_special_tokens=True)
            
        with torch.no_grad():
            outputs = self.model(
                input_ids=inputs["input_ids"].to(device),
                attention_mask=inputs["attention_mask"].to(device)
            )
        logits = outputs.logits
        first_position_logits = logits[0, len(inputs["input_ids"][0]) - 1, :]
        probs = F.softmax(first_position_logits, dim=-1)

        # ######        
        # max_prob, token_id = torch.max(probs, dim=-1)
        # print(f'Token ID: {token_id.item()}->Token: {self.tokenizer.convert_ids_to_tokens(token_id.item())} Probability: {max_prob.item()} || A id: {self.tokenizer.convert_tokens_to_ids("A")} || B id: {self.tokenizer.convert_tokens_to_ids("B")}')
        # ######
        
        output_prob = [0, 0]

        answer_tokens = ["A", "B"]
        for posi, tok in enumerate(answer_tokens):
            output_prob[posi] = probs[self.tokenizer.convert_tokens_to_ids(tok)].item()

        return [output_prob]


