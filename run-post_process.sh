#!/bin/sh

python post_process.py \
    --csv_path completions/gpt-j-6b-tldr/greedy.csv \
    --save_path completions/gpt-j-6b-tldr/greedy.csv

python post_process.py \
    --csv_path completions/gpt-j-6b-tldr/test-beam6-1.csv \
    --save_path completions/gpt-j-6b-tldr/test-beam6-1.csv

python post_process.py \
    --csv_path completions/gpt-j-6b-tldr/test-beam6-2.csv \
    --save_path completions/gpt-j-6b-tldr/test-beam6-2.csv

python post_process.py \
    --csv_path completions/gpt-j-6b-tldr/test-temp5-1.csv \
    --save_path completions/gpt-j-6b-tldr/test-temp5-1.csv

python post_process.py \
    --csv_path completions/gpt-j-6b-tldr/test-temp5-2.csv \
    --save_path completions/gpt-j-6b-tldr/test-temp5-2.csv