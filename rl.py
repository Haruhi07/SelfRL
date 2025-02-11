import torch
import shutil
import argparse

from accelerate import Accelerator
from datasets import load_from_disk
from transformers import (
    AutoModelForSeq2SeqLM,
    TrainingArguments,
    AutoTokenizer,
    HfArgumentParser,
)

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Literal, Optional

from trl import (
    DPOConfig,
    DPOTrainer,
    ModelConfig,
    ScriptArguments,
    TrlParser,
    get_kbit_device_map,
    get_peft_config,
    get_quantization_config,
)
from trl.trainer.utils import SIMPLE_CHAT_TEMPLATE


def get_labeled_dataset(dataset_path):
    score_train_dataset = load_from_disk(f"{dataset_path}/train")#.select(range(100))
    score_test_dataset = load_from_disk(f"{dataset_path}/test")#.select(range(100))
    def set_chosen_and_rejected(data):
        if data["score1"] > data["score2"]:
            chosen = data["completion1"]
            rejected = data["completion2"]
        else:
            chosen = data["completion2"]
            rejected = data["completion1"]
        return {
            "prompt": data["query"],
            "chosen": chosen,
            "rejected": rejected
        }
    
    train_dataset = score_train_dataset.map(
        set_chosen_and_rejected, 
        num_proc=4, 
        remove_columns=score_train_dataset.column_names
    )
    test_dataset = score_test_dataset.map(
        set_chosen_and_rejected, 
        num_proc=4, 
        remove_columns=score_test_dataset.column_names
    )
    return train_dataset, test_dataset


def make_parser(subparsers: argparse._SubParsersAction = None):
    dataclass_types = (ScriptArguments, DPOConfig, ModelConfig)
    if subparsers is not None:
        parser = subparsers.add_parser("dpo", help="Run the DPO training script", dataclass_types=dataclass_types)
    else:
        parser = TrlParser(dataclass_types)
    return parser


if __name__ == "__main__":
    parser = make_parser()
    script_args, training_args, model_args = parser.parse_args_and_config()
    # remove output_dir if exists
    shutil.rmtree(training_args.output_dir, ignore_errors=True)

    ################
    # Model & Tokenizer
    ################
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_args.model_name_or_path,
        device_map={"": Accelerator().local_process_index}
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        trust_remote_code=model_args.trust_remote_code,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.chat_template is None:
        tokenizer.chat_template = SIMPLE_CHAT_TEMPLATE
    if script_args.ignore_bias_buffers:
        # torch distributed hack
        model._ddp_params_and_buffers_to_ignore = [
            name for name, buffer in model.named_buffers() if buffer.dtype == torch.bool
        ]

    ################
    # Dataset
    ################
    if script_args.score_dataset:
        train_dataset, test_dataset = get_labeled_dataset(script_args.dataset_name)
    else:
        train_dataset = load_from_disk(f"{script_args.dataset_name}/train")
        test_dataset = load_from_disk(f"{script_args.dataset_name}/test")

    ################
    # Training
    ################
    trainer = DPOTrainer(
        model=model,
        ref_model=None,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        processing_class=tokenizer,
    )
    trainer.train()

    if training_args.eval_strategy != "no":
        metrics = trainer.evaluate()
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    # Save and push to hub
    trainer.save_model(training_args.output_dir)