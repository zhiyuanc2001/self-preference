WORK_DIR="Your code path"
DATA_DIR="data to the alpacaeval dataset"
DATA_TYPE="alpaca_eval"

# Model_type is the judge model
MODEL_TYPE="/data/hf_models/Llama/Llama-3.1-70B-Instruct"
MODEL_NAME_1="Llama-3.1-70B-Instruct"
MODEL_NAME_2="Qwen2.5-0.5B-Instruct"
python ${WORK_DIR}/gen_preference.py \
    --data_dir ${DATA_DIR} \
    --data_type ${DATA_TYPE} \
    --model_name_1 ${MODEL_NAME_1} \
    --model_name_2 ${MODEL_NAME_2} \
    --model_type ${MODEL_TYPE} \
    --is_instruct \
    --batch_size 1

python ${WORK_DIR}/gen_preference.py \
    --data_dir ${DATA_DIR} \
    --data_type ${DATA_TYPE} \
    --model_name_1 ${MODEL_NAME_2} \
    --model_name_2 ${MODEL_NAME_1} \
    --model_type ${MODEL_TYPE} \
    --is_instruct \
    --batch_size 1


python ${WORK_DIR}/average_preference.py \
    --evaluation_type ${DATA_TYPE} \
    --evaluator_name $(basename "$MODEL_TYPE") \
    --model_name_1 ${MODEL_NAME_1} \
    --model_name_2 ${MODEL_NAME_2}