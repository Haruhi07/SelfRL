#!/bin/sh

export CUDA_VISIBLE_DEVICES=1
export STRATEGY=summac

python bart_sampling_with_different_strategy.py \
    --save_path completions/bart-xsum-dpo/$STRATEGY-beam6-1+greedy.csv \
    --model_name_or_path bart-xsum-dpo-beam6-1+greedy/$STRATEGY/checkpoint-11000 \
    --dataset_name_or_path EdinburghNLP/xsum \
    --split test \
    --max_doc_length 512 \
    --max_new_tokens 63 \
    --batch_size 8 \
    --decoding_strategy beam6 \
    --num_return_sequences 1
