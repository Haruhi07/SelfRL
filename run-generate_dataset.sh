#!/bin/sh

export CUDA_VISIBLE_DEVICES=1

python generate_dataset.py \
    --save_path completions/xsum/bart-large-xsum/test-greedy.csv \
    --model_name_or_path facebook/bart-large-xsum \
    --dataset_name_or_path EdinburghNLP/xsum \
    --split test \
    --max_doc_length 512 \
    --max_new_tokens 63 \
    --batch_size 64 \
    --decoding_strategy greedy \
    --num_return_sequences 1
