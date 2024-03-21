'''

Modified from https://github.com/yizhongw/Tk-Instruct/blob/main/src/convert_data_to_s2s.py

This script is used to load raw data, perform train/dev/test split, and then save splitted data to disk.

'''


import sys
sys.path.insert(1, 'Tk-Instruct/src')
from ni_collator import DataCollatorForNI
from utils import train_dev_test_split_by_task
from typing import Optional
import os
import json
import glob
import tqdm
import pandas as pd
import importlib
from transformers import HfArgumentParser, GPT2TokenizerFast
# from run_s2s import DataTrainingArguments
from datasets import load_dataset, DatasetDict, Dataset, load_from_disk
from dataclasses import dataclass, field
from nltk import sent_tokenize

@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """
    lang: str = field(default=None, metadata={"help": "Language id for multilingual model."})
    data_dir: str = field(
        default=None, metadata={"help": "The directory for saving the NaturalInstructions train/dev/test splits."}
    )
    task_dir: str = field(
        default=None, metadata={"help": "The directory for saving the NaturalInstructions tasks json files."}
    )
    task_split_file_name: Optional[str] = field(
        default="cl_dialogue_tasks",
        metadata={"help": "The task split txt name."}
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    max_source_length: Optional[int] = field(
        default=1024,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    max_target_length: Optional[int] = field(
        default=128,
        metadata={
            "help": "The maximum total sequence length for target text after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    pad_to_max_length: bool = field(
        default=False,
        metadata={
            "help": "Whether to pad all samples to model maximum sentence length. "
            "If False, will pad the samples dynamically when batching to the maximum length in the batch. More "
            "efficient on GPU but very bad for TPU."
        },
    )
    max_num_instances_per_task: int = field(
        default=None, metadata={"help": "The maximum number of instances we will consider for each training task."}
    )
    max_num_instances_per_eval_task: int = field(
        default=500, metadata={"help": "The maximum number of instances we will consider for each validation/test task."}
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
            "value if set."
        },
    )
    max_predict_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of prediction examples to this "
            "value if set."
        },
    )
    num_beams: Optional[int] = field(
        default=None,
        metadata={
            "help": "Number of beams to use for evaluation. This argument will be passed to ``model.generate``, "
            "which is used during ``evaluate`` and ``predict``."
        },
    )
    ignore_pad_token_for_loss: bool = field(
        default=True,
        metadata={
            "help": "Whether to ignore the tokens corresponding to padded labels in the loss computation or not."
        },
    )
    source_prefix: Optional[str] = field(
        default="", metadata={"help": "A prefix to add before every source text (useful for T5 models)."}
    )

    forced_bos_token: Optional[str] = field(
        default=None,
        metadata={
            "help": "The token to force as the first generated token after the decoder_start_token_id."
            "Useful for multilingual models like mBART where the first generated token"
            "needs to be the target language token (Usually it is the target language token)"
        },
    )
    add_task_name: Optional[bool] = field(
        default=False,
        metadata={"help": "whether to preappend task name before the task input."}
    )
    add_task_definition: Optional[bool] = field(
        default=True,
        metadata={"help": "whether to preappend task definition before the task input."}
    )
    num_pos_examples: Optional[int] = field(
        default=0,
        metadata={"help": "number of in-context positive examples."}
    )
    num_neg_examples: Optional[int] = field(
        default=0,
        metadata={"help": "number of in-context negative examples."}
    )
    add_explanation: Optional[bool] = field(
        default=False,
        metadata={"help": "whether to add explanation for both the postive examples and negtive examples."}
    )
    tk_instruct: Optional[bool] = field(
        default=False,
        metadata={"help": "tk_instruct will train a model combining all valid instruction encodings. This will overwrite the other settings about instruction encoding."} 
    )
    
    def __post_init__(self):
        pass

@dataclass
class CustomizedArguments:
    output_dir: str = field(
        default="data/text2text/", metadata={"help": "The directory for saving splits."}
    )
    output_dir_for_official_test: str = field(
        default="data/text2text/", metadata={"help": "The directory for saving splits."}
    )
    seed: Optional[int] = field(
        default=10,
        metadata={"help": "random seed"}
    )
    save_official_test_tasks: Optional[bool] = field(
        default=False,
        metadata={"help": "If True, will also save the official test tasks data in the disk."} 
    )



if __name__ == "__main__":
    parser = HfArgumentParser((DataTrainingArguments, CustomizedArguments))
    args, customized_args = parser.parse_args_into_dataclasses()

    # load data from natural-instructions
    raw_datasets = load_dataset(
        # "Tk-Instruct/src/ni_dataset.py",
        "continual_learning/ni_dataset_for_cl.py", # use modified dadatset script
        data_dir=args.data_dir, 
        task_dir=args.task_dir, 
        max_num_instances_per_task=args.max_num_instances_per_task,
        max_num_instances_per_eval_task=args.max_num_instances_per_eval_task,
        task_split_file_name=args.task_split_file_name
    )

    print(f"raw_datasets: {raw_datasets}")

    train_instances, dev_instances, test_instances = train_dev_test_split_by_task(raw_datasets, 
        args.max_num_instances_per_task, args.max_num_instances_per_eval_task)

    print(f"train: {train_instances[0]['id']}")
    print(f"dev: {dev_instances[0]['id']}")
    print(f"test: {test_instances[0]['id']}")

    # sys.exit(0)

    # save splitted data to disk for later usage
    train_data_save_path = f"{customized_args.output_dir}/train"
    dev_data_save_path = f"{customized_args.output_dir}/dev"
    test_data_save_path = f"{customized_args.output_dir}/test"
    
    # Dataset.from_list(train_instances).save_to_disk(train_data_save_path)
    # Dataset.from_list(dev_instances).save_to_disk(dev_data_save_path)
    # Dataset.from_list(test_instances).save_to_disk(test_data_save_path)

    # # also save the official test tasks to disk 
    # if customized_args.save_official_test_tasks:
    #     raw_datasets['test'].save_to_disk(customized_args.output_dir_for_official_test)
        

    # # try to load the dataset from disk
    # train_dataset = load_from_disk(train_data_save_path)
    # dev_dataset = load_from_disk(dev_data_save_path)
    # test_dataset = load_from_disk(test_data_save_path)

    # print(f"train: {train_dataset[0]['id']} {len(train_dataset)}")
    # print(f"dev: {dev_dataset[0]['id']} {len(dev_dataset)}")
    # print(f"test: {test_dataset[0]['id']} {len(test_dataset)}")

    # if customized_args.save_official_test_tasks:
    #     official_test_dataset = load_from_disk(customized_args.output_dir_for_official_test)
    #     print(f"official test: {official_test_dataset[0]['id']} {len(official_test_dataset)}")
    
        