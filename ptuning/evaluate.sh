PRE_SEQ_LEN=64
STEP=3000
NUM_GPUS=1

#torchrun --standalone --nnodes=1 --nproc-per-node=$NUM_GPUS main.py \
python main.py \
    --do_predict \
    --test_file ../data/dev.json \
    --overwrite_cache \
    --prompt_column instruction \
    --response_column output \
    --ptuning_checkpoint ./output/ner/model1/checkpoint-$STEP \
    --output_dir ./output/ner/model1/checkpoint-$STEP \
    --overwrite_output_dir \
    --max_source_length 352 \
    --max_target_length 200 \
    --per_device_eval_batch_size 8 \
    --predict_with_generate \
    --pre_seq_len $PRE_SEQ_LEN \




#    --max_source_length 512 \
#    --max_target_length 200 \