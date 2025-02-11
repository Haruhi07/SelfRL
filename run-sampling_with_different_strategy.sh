#!/bin/sh

export CUDA_VISIBLE_DEVICES=0

python sampling_with_different_strategy.py \
    --save_path completions/tldr/gpt-j-6b/test-beam6-1.csv \
    --decoding_strategy beam6 \
    --num_return_sequences 1 \
    --model_name_or_path CarperAI/openai_summarize_tldr_sft \
    --tokenizer_name_or_path EleutherAI/gpt-j-6b \
    --dataset_name_or_path CarperAI/openai_summarize_tldr \
    --split test \
    --max_doc_length 2000 \
    --max_new_tokens 50 \
    --batch_size 1
