#!/bin/sh

export CUDA_VISIBLE_DEVICES=1
export DATASET_PATH=completions/xsum/bart-large-xsum

python score_dataset.py \
    --raw_dataset_path1 $DATASET_PATH/train-beam6-1.csv \
    --raw_dataset_path2 $DATASET_PATH/train-greedy.csv \
    --output_path $DATASET_PATH/beam6-1+greedy/summac/train