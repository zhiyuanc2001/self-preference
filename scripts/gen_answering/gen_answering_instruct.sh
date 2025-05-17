WORK_DIR="Your code path"

python ${WORK_DIR}/gen_answering.py \
    --data_dir ${WORK_DIR}/datasets/alpaca_eval \
    --model_type Your post-trained model path \
    --is_instruct \
    --batch_size 1