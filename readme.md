## Overview
This is the anonymized version code for our paper *Beyond the Surface: Measuring Self-Preference in LLM Judgments*.



## Data

### Instruction Data
The instructions used for each dataset are in the `instruction_fullset/` folder.


### Preference Data
We have uploaded the judge model preference results in `model_preferences_fullset.tar.gz`. Please use 
```bash
tar -zxf model_preferences_fulset.tar.gz
```
to unzip the `.tar.gz` file.

The file structure is as follows:
```
├── /model_preferences_fullset/
│  ├── /alpaca_eval_500_id_wr/
│  │  ├── /evaluator_Llama-3.1-8B/
│  │  │  ├── average_Llama-3.1-8B_Llama-3.1-8B-Instruct.jsonl
│  │  │  ├── ...
│  │  ├── /evaluator_deepseek-v3_golden
│  │  │  ├── merge_Llama-3.1-8B_Llama-3.1-8B-Instruct.jsonl
│  │  │  ├── ...
│  │  ├── ...
│  ├── /translation_500_id_wr/
│  ├── /truthfulness_500_id_wr/
```
Here are some descriptions for the `model_preferences_fullset/` directory:  
- `alapca_eval_500_id_wr/`, `truthfulness_500_id_wr/`, and `translation_500_id_wr/`: These directories represent the judgment results of the judge models (including gold judges) on the model responses for the AlpacaEval (helpfulness), AlpacaEval (truthfulness), and WMT-19 datasets, respectively.
- `evaluator_*/`: The evaluation results of each judge model. For example: `evaluator_deepseek-v3_golden/`, `evaluator_gemini-flash-1.5_golden/`, and `evaluator_gpt-4o-mini_golden/` is the gold judgments. `evaluator_Llama-3.1-8B/` is the judgment results of *Llama-3.1-8B model* （For ease of description and to distinguish these models from gold judges, we refer to them as **normal judges**）. 
- File description:
    - For files under normal judges `evalulator_normal_judge/average_model1_model2.jsonl` (e.g., `evaluator_Llama-3.1-8B/average_Llama-3.1-8B_Llama-3.1-8B-Instruct.jsonl`), the two models `model1, model2` in the file name (e.g., Llama-3.1-8B and Llama-3.1-8B-Instruct) represent the two models that generate responses, where the judge model `normal_judge`(e.g., Llama-3.1-8B) needs to evaluate the responses of these two models. The data in each line of the file is as follows:
    ```jsonl
    {"id": xxx, "model1_response": xxx, "model2_response", "preferences": xxx}
    ```
    where `preferences` refers to the preferred response by the judge model after accounting for swapped positions of the responses (for mitigating position bias).
    - For files under gold judges `evaluator_gold_judge/merge_model1_model2.jsonl` (e.g., `evaluator_deepseek-v3_golden/merge_Llama-3.1-8B_Llama-3.1-8B-Instruct.jsonl`), the two models `model1, model2` in the file name (e.g., Llama-3.1-8B and Llama-3.1-8B-Instruct) represent the two models that generate responses, where the `gold judge` (e.g., deepseek-v3) needs to evaluate the responses of these two models. The data in each line of the file is as follows:
    ```jsonl
    {"id": xxx, "model1_response": xxx, "model2_response", "preferences": [xxx, yyy]}
    ```
    where `preferences` is a list, which refers to the preferred responses by the gold judge **before and after** swapping the positions of the responses.


## Response Generation
### Installation

Install Package (python>=3.9)
```bash
pip install -r requirements.txt
```


### Response Generation
#### AlpacaEval
Generate model responses for AlpacaEval(helpfulness) by `gen_answering.py`

For pre-trained model:
```bash
WORK_DIR="Your code path"

python ${WORK_DIR}/gen_answering.py \
    --data_dir datasets/alpaca_eval \
    --model_type Your pre-trained model \
    --batch_size 1
```

For post-trained model:
```bash
WORK_DIR="Your code path"

python ${WORK_DIR}/gen_answering.py \
    --data_dir ${WORK_DIR}/datasets/alpaca_eval \
    --model_type Your post-trained model \
    --is_instruct \
    --batch_size 1
```

#### TruthfulQA
Generate model responses for AlpacaEval(truthfulness) by `gen_truthfulness.py`  

For pre-trained model:
```bash
WORK_DIR="Your code path"

python ${WORK_DIR}/gen_truthfulness.py \
    --data_dir datasets/truthful_qa \
    --model_type Your pre-trained model \
    --batch_size 1
```


For post-trained model:
```bash
WORK_DIR="Your code path"

python ${WORK_DIR}/gen_truthfulness.py \
    --data_dir datasets/truthful_qa \
    --model_type Your post-trained model \
    --is_instruct \
    --batch_size 1
```


#### WMT
Generate model responses for WMT by `gen_translation.py`  

For pre-trained model:
```bash
WORK_DIR="Your code path"

python ${WORK_DIR}/gen_translation.py \
    --data_dir datasets/wmt_de-en \
    --model_type Your pre-trained model \
    --batch_size 1
```

For post-trained model:
```bash
WORK_DIR="Your code path"

python ${WORK_DIR}/gen_translation.py \
    --data_dir datasets/wmt_de-en \
    --model_type Your post-trained model \
    --is_instruct \
    --batch_size 1
```


### Preference Generation
Generate model preference by `gen_preference.py`


For normal judge:
```bash
WORK_DIR="Your code path"
DATA_DIR="data to the directory for saving model responses"
# For example: "model_responses_fullset/question_answering/alpaca_eval"


DATA_TYPE="alpaca_eval"
# DATA_TYPE="truthfulness"
# DATA_TYPE="translation"

MODEL_TYPE="judge model type"

# model1 and model2 for example:
MODEL_NAME_1="Qwen2.5-7B"
MODEL_NAME_2="Llama3.1-8B"

python ${WORK_DIR}/gen_preference.py \
    --data_dir ${DATA_DIR} \
    --data_type ${DATA_TYPE} \
    --model_name_1 ${MODEL_NAME_1} \
    --model_name_2 ${MODEL_NAME_2} \
    --model_type ${MODEL_TYPE} \
#    `--instruct` if use post-trained model \
    --batch_size 1

python ${WORK_DIR}/gen_preference.py \
    --data_dir ${DATA_DIR} \
    --data_type ${DATA_TYPE} \
    --model_name_1 ${MODEL_NAME_2} \
    --model_name_2 ${MODEL_NAME_1} \
    --model_type ${MODEL_TYPE} \
#    `--instruct` if use post-trained model \
    --batch_size 1

python ${WORK_DIR}/average_preference.py \
    --evaluation_type ${DATA_TYPE} \
    --evaluator_name $(basename "$MODEL_TYPE") \
    --model_name_1 ${MODEL_NAME_1} \
    --model_name_2 ${MODEL_NAME_2}
```

For gold judge model:
```bash
WORK_DIR="Your code path"

DATA_DIR="data to the directory for saving model responses"
# For example "model_responses_fullset/question_answering/alpaca_eval"

DATA_TYPE="alpaca_eval"
# DATA_TYPE="truthfulness"
# DATA_TYPE="translation"

MODEL_TYPE="gold judge type (e.g., gemini-flash-1.5, gpt-4o-mini, deepseek-v3)"

# model1 and model2:
MODEL_NAME_1="Llama-3.1-70B"
MODEL_NAME_2="Llama-3.1-8B"

python ${WORK_DIR}/gen_preference.py \
    --data_dir ${DATA_DIR} \
    --data_type ${DATA_TYPE} \
    --model_name_1 ${MODEL_NAME_1} \
    --model_name_2 ${MODEL_NAME_2} \
    --model_type ${MODEL_TYPE} \
    --is_instruct \
    --use_infer_generate \
    --batch_size 5

python ${WORK_DIR}/gen_preference.py \
    --data_dir ${DATA_DIR} \
    --data_type ${DATA_TYPE} \
    --model_name_1 ${MODEL_NAME_2} \
    --model_name_2 ${MODEL_NAME_1} \
    --model_type ${MODEL_TYPE} \
    --is_instruct \
    --use_infer_generate \
    --batch_size 5

python ${WORK_DIR}/merge_preference_infer_generate.py \
    --evaluation_type ${DATA_TYPE} \
    --evaluator_name $(basename "$MODEL_TYPE") \
    --model_name_1 ${MODEL_NAME_1} \
    --model_name_2 ${MODEL_NAME_2}
```

