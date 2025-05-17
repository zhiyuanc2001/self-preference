WORK_DIR="Your code path"

python ${WORK_DIR}/gen_answering.py \
    --data_dir datasets/alpaca_eval \
    --model_type Your pre-trained path \
    --batch_size 1