WORK_DIR="Your code path"
DATA_DIR="data to the alpacaeval dataset"
DATA_TYPE="alpaca_eval"


# gold judge
MODEL_TYPE="claude-3.5-haiku"

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