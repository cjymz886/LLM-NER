



python fine-tune.py  \
    --data_path ./data/pkumod-ccks_query_list_train4.txt \
    --output_dir output/m_ctx \
    --num_train_epochs 4 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --save_strategy epoch \
    --learning_rate 5e-5 \
    --lr_scheduler_type cosine \
    --adam_beta1 0.9 \
    --adam_beta2 0.98 \
    --adam_epsilon 1e-8 \
    --max_grad_norm 1.0 \
    --weight_decay 1e-4 \
    --warmup_ratio 0.0 \
    --logging_steps 100 \
    --gradient_checkpointing True \
    --bf16 True \
    --tf32 True
