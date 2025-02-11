#!/bin/sh

export CUDA_VISIBLE_DEVICES=1

python rl.py \
    --score_dataset \
    --output_dir bart-xsum-dpo-beam6-1+temp5-1/sbert+summac \
    --dataset_name completions/xsum/bart-large-xsum/beam6-1+temp5-1/sbert+summac \
    --run_name bart-xsum-dpo-beam6-1+temp5-1-sbert+summac \
    --logging_dir bart-xsum-dpo-beam6-1+temp5-1/log/sbert+summac \
    --model_name_or_path facebook/bart-large-xsum \
    --learning_rate 1.0e-6 \
    --num_train_epochs 2 \
    --save_total_limit 3 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 32 \
    --gradient_checkpointing \
    --logging_steps 100 \
    --eval_strategy steps \
    --eval_steps 500 \
    --no_remove_unused_columns \
    --report_to tensorboard \
    --max_prompt_length 512 \
    --max_completion_length 62 
