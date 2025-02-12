#!/bin/sh

python post_process.py \
    --csv_path completions/tldr/gpt-j-6b/test-greedy.csv \
    --save_path completions/tldr/gpt-j-6b/test-greedy.csv

python post_process.py \
    --csv_path completions/tldr/gpt-j-6b/test-beam6-1.csv \
    --save_path completions/tldr/gpt-j-6b/test-beam6-1.csv

python post_process.py \
    --csv_path completions/tldr/gpt-j-6b/test-beam6-2.csv \
    --save_path completions/tldr/gpt-j-6b/test-beam6-2.csv

python post_process.py \
    --csv_path completions/tldr/gpt-j-6b/test-temp5-1.csv \
    --save_path completions/tldr/gpt-j-6b/test-temp5-1.csv

python post_process.py \
    --csv_path completions/tldr/gpt-j-6b/test-temp5-2.csv \
    --save_path completions/tldr/gpt-j-6b/test-temp5-2.csv