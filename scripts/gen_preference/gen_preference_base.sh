WORK_DIR="Your code path"
DATA_DIR="data to the alpacaeval dataset"
DATA_TYPE="alpaca_eval"


MODEL_TYPE="/data/hf_models/Qwen/Qwen2.5-7B"
MODEL_NAME_1="Qwen2.5-7B"
MODEL_NAME_2="Llama3.1-8B"

python ${WORK_DIR}/gen_preference.py \
    --data_dir ${DATA_DIR} \
    --data_type ${DATA_TYPE} \
    --model_name_1 ${MODEL_NAME_1} \
    --model_name_2 ${MODEL_NAME_2} \
    --model_type ${MODEL_TYPE} \
    --batch_size 1

python ${WORK_DIR}/gen_preference.py \
    --data_dir ${DATA_DIR} \
    --data_type ${DATA_TYPE} \
    --model_name_1 ${MODEL_NAME_2} \
    --model_name_2 ${MODEL_NAME_1} \
    --model_type ${MODEL_TYPE} \
    --batch_size 1


python ${WORK_DIR}/average_preference.py \
    --evaluation_type ${DATA_TYPE} \
    --evaluator_name $(basename "$MODEL_TYPE") \
    --model_name_1 ${MODEL_NAME_1} \
    --model_name_2 ${MODEL_NAME_2}