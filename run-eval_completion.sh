#!/bin/sh

export CUDA_VISIBLE_DEVICES=1

python eval-completion.py \
    --completion_path completions/bart-xsum-dpo