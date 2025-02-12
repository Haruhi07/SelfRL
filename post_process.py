import torch
from torch.utils.data import DataLoader

import transformers
from transformers import (
    HfArgumentParser,
)

import os
import pandas as pd
from tqdm import tqdm
from dataclasses import dataclass

@dataclass
class ScriptArgs:
    csv_path: str
    save_path: str


def post_process(row):
    source = row["query"]
    generation = row["completion"]
    summary = generation[len(source): ]
    return summary

if __name__ == "__main__":
    parser = HfArgumentParser((ScriptArgs))
    script_args = parser.parse_args_into_dataclasses()[0]

    df = pd.read_csv(script_args.csv_path)
    summaries = df.apply(post_process, axis="columns")
    df["completion"] = [summary.strip() for summary in summaries]
    df.to_csv(script_args.save_path)
