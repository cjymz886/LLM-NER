PRE_SEQ_LEN=64
LR=2e-2
NUM_GPUS=1

python main.py \
    --do_train \
    --train_file ../data/train.json \
    --preprocessing_num_workers 10 \
    --prompt_column instruction \
    --response_column output \
    --overwrite_cache \
    --output_dir output/ner/model1 \
    --overwrite_output_dir \
    --max_source_length 350 \
    --max_target_length 200 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 8 \
    --gradient_accumulation_steps 1 \
    --predict_with_generate \
    --max_steps 3000 \
    --logging_steps 10 \
    --save_steps 1000 \
    --learning_rate $LR \
    --pre_seq_len $PRE_SEQ_LEN \
