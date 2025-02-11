import torch
from torch.utils.data import DataLoader

import datasets
from datasets import load_dataset

import transformers
from transformers import (
    AutoModelForCausalLM, 
    HfArgumentParser,
    TrainingArguments,
    AutoTokenizer,
    GenerationConfig,
    BitsAndBytesConfig
)

import os
import pandas as pd
from tqdm import tqdm
from dataclasses import dataclass

@dataclass
class ScriptArgs:
    model_name_or_path: str
    dataset_name_or_path: str
    split: str
    max_doc_length: int
    max_new_tokens: int
    batch_size: int
    decoding_strategy: str
    save_path: str
    tokenizer_name_or_path: str
    num_return_sequences: int = 1


dm_single_close_quote = "\u2019"  # unicode
dm_double_close_quote = "\u201d"
END_TOKENS = [
    ".",
    "!",
    "?",
    "...",
    "'",
    "`",
    '"',
    dm_single_close_quote,
    dm_double_close_quote,
    ")",
]  # acceptable ways to end a sentence

def fix_missing_period(line):
    """Adds a period to a line that is missing a period"""
    if "@highlight" in line:
        return line
    if line == "":
        return line
    if line[-1] in END_TOKENS:
        return line
    # print line[-1]
    return line + "."


def process_dataset(dataset, tokenizer, max_doc_length):
    def tokenize_fn(data):
        doc = data["prompt"]
        ref = data["label"]
        query = doc
        inputs = tokenizer(query, truncation=True)
        label = tokenizer(ref)["input_ids"]
        return {
            "source": doc,
            "query": query,
            "reference": ref,
            "input_ids": inputs["input_ids"],
            "attention_mask": inputs["attention_mask"],
            "label": label
        }
    dataset = dataset.map(tokenize_fn, num_proc=4)
    return dataset


def get_model_name(model_name_or_path):
    return model_name_or_path.split("/")[1]


def get_gen_config(decoding_strategy, tokenizer):
    gen_config = None
    if decoding_strategy == "greedy":
        gen_config = GenerationConfig(
            min_length = 10,
            max_new_tokens = script_args.max_new_tokens,
            do_sample=False,
            num_beams=1,
            pad_token_id=tokenizer.eos_token_id
        )
        print(f"Using greedy decoding strategy!")
    elif decoding_strategy == "temp5":
        gen_config = GenerationConfig(
            min_length = 10,
            max_new_tokens = script_args.max_new_tokens,
            do_sample=True,
            temperature=5.0,
            pad_token_id=tokenizer.eos_token_id,
            num_return_sequences=script_args.num_return_sequences,
        )
        print(f"Using random sampling with temprature 5 and return {script_args.num_return_sequences} sequences!")
    elif decoding_strategy == "beam6":
        gen_config = GenerationConfig(
            max_new_tokens = script_args.max_new_tokens,
            early_stopping=True,
            num_beams=6,
            pad_token_id=tokenizer.eos_token_id,
            num_return_sequences=script_args.num_return_sequences,
        )
        print(f"Using beam search with num_beam 6 and return {script_args.num_return_sequences} sequences!")
    if gen_config is None:
        raise ValueError("Unspecified decoding strategy!")
    return gen_config

if __name__ == "__main__":
    parser = HfArgumentParser((ScriptArgs))
    script_args = parser.parse_args_into_dataclasses()[0]

    ##########
    # Prepare model and dataset
    ##########

    lm = AutoModelForCausalLM.from_pretrained(
        script_args.model_name_or_path, 
        device_map="auto",
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(script_args.tokenizer_name_or_path, use_fast=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    dataset = load_dataset(script_args.dataset_name_or_path, split=script_args.split)
    prompted_dataset = process_dataset(dataset, tokenizer, script_args.max_doc_length)
    dataset = prompted_dataset.remove_columns(dataset.column_names + ["query"])

    def collate_fn(batch):
        pad_token_id = tokenizer.pad_token_id

        input_ids = [example["input_ids"] for example in batch]
        attention_masks = [example["attention_mask"] for example in batch]
        
        # Find the maximum length in the batch
        max_doc_length = max(len(ids) for ids in input_ids)
        
        # Pad input_ids, attention_masks and labels to the corresponding max_length
        padded_input_ids = torch.tensor(
            [[pad_token_id] * (max_doc_length - len(ids)) + ids for ids in input_ids], dtype=torch.long
        )
        padded_attention_masks = torch.tensor(
            [[0] * (max_doc_length - len(mask)) + mask for mask in attention_masks], dtype=torch.long
        )
        
        return {
            "input_ids": padded_input_ids,
            "attention_mask": padded_attention_masks,
        }
    test_loader = DataLoader(dataset, batch_size=script_args.batch_size, collate_fn=collate_fn)

    ##########
    # Generate completion
    ##########
    generation_input_name = ["input_ids", "attention_mask"]
    completion = []
    gen_config = get_gen_config(script_args.decoding_strategy, tokenizer)

    for batch in tqdm(test_loader):
        input_dict = {k: v.to(lm.device) for k, v in batch.items() if k in generation_input_name}
        generation = lm.generate(**input_dict, generation_config=gen_config)
        # Remove prompt from the generation
        prompt_length = input_dict['input_ids'].shape[1]
        batch_completion = generation[script_args.num_return_sequences-1::script_args.num_return_sequences, :]
        completion.extend(tokenizer.batch_decode(batch_completion, skip_special_tokens=True))

    ##########
    # Save completion to completions/{model}/{strategy}.csv
    ##########
    df = pd.DataFrame({
        "query": prompted_dataset["query"],
        "completion": completion,
        "source": prompted_dataset["query"],
        "reference": prompted_dataset["reference"],
    })
    df.to_csv(script_args.save_path)