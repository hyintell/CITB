#!/usr/bin/env python
# coding=utf-8
# Copyright 2021 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Fine-tuning the library models for sequence to sequence.
"""
# You can also adapt this script on your own sequence to sequence task. Pointers for this are left as comments.

import random
import logging
import os
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional
import sys
sys.path.insert(1, 'Tk-Instruct/src')
from ni_collator import DataCollatorForNI
# from ni_trainer import NITrainer, DenserEvalCallback, EarlyStoppingCallback
from ni_trainer_for_cl import NITrainer, DenserEvalCallback, EarlyStoppingCallback
from utils import train_dev_test_split_by_task, get_replay_instances_by_task
from seq2seqCL import Seq2SeqCL

import torch
import json
import datasets
import nltk  # Here to have a nice missing dependency error message early on
import numpy as np
from datasets import load_dataset, load_metric, DatasetDict, Dataset, load_from_disk
from compute_metrics import compute_metrics, compute_grouped_metrics
from torch.utils.data.dataloader import DataLoader

# import evaluate
import transformers
from filelock import FileLock
from transformers import (
    AutoConfig,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    HfArgumentParser,
    MBart50Tokenizer,
    MBart50TokenizerFast,
    MBartTokenizer,
    MBartTokenizerFast,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import check_min_version, is_offline_mode, send_example_telemetry
from transformers.utils.versions import require_version

# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
# check_min_version("4.26.0.dev0")

require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/summarization/requirements.txt")

logger = logging.getLogger(__name__)

try:
    nltk.data.find("tokenizers/punkt")
except (LookupError, OSError):
    if is_offline_mode():
        raise LookupError(
            "Offline mode: run this script without TRANSFORMERS_OFFLINE first to download nltk data files"
        )
    with FileLock(".lock") as lock:
        nltk.download("punkt", quiet=True)

# A list of all multilingual tokenizer which require lang attribute.
MULTILINGUAL_TOKENIZERS = [MBartTokenizer, MBartTokenizerFast, MBart50Tokenizer, MBart50TokenizerFast]


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": (
                "Will use the token generated when running `huggingface-cli login` (necessary to use this script "
                "with private models)."
            )
        },
    )
    resize_position_embeddings: Optional[bool] = field(
        default=None,
        metadata={
            "help": (
                "Whether to automatically resize the position embeddings if `max_source_length` exceeds "
                "the model's position embeddings."
            )
        },
    )


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
    data_dir_for_task_order: str = field(
        default=None, metadata={"help": "The directory to task orders."}
    )
    data_dir_for_official_test: str = field(
        default=None, metadata={"help": "The directory for the official test tasks data."}
    )
    data_dir_for_initial_training_dir: str = field(
        default=None, metadata={"help": "The directory for the inital multi-task learning tasks data."}
    )
    max_num_instances_per_task: int = field(
        default=None, metadata={"help": "The maximum number of instances we will consider for each training task."}
    )
    max_num_instances_per_eval_task: int = field(
        default=500, metadata={"help": "The maximum number of instances we will consider for each validation/test task."}
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
            "help": (
                "The maximum total input sequence length after tokenization. Sequences longer "
                "than this will be truncated, sequences shorter will be padded."
            )
        },
    )
    max_target_length: Optional[int] = field(
        default=128,
        metadata={
            "help": (
                "The maximum total sequence length for target text after tokenization. Sequences longer "
                "than this will be truncated, sequences shorter will be padded."
            )
        },
    )
    pad_to_max_length: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether to pad all samples to model maximum sentence length. "
                "If False, will pad the samples dynamically when batching to the maximum length in the batch. More "
                "efficient on GPU but very bad for TPU."
            )
        },
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of training examples to this "
                "value if set."
            )
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
                "value if set."
            )
        },
    )
    max_predict_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of prediction examples to this "
                "value if set."
            )
        },
    )
    num_beams: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "Number of beams to use for evaluation. This argument will be passed to ``model.generate``, "
                "which is used during ``evaluate`` and ``predict``."
            )
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
            "help": (
                "The token to force as the first generated token after the decoder_start_token_id."
                "Useful for multilingual models like mBART where the first generated token"
                "needs to be the target language token (Usually it is the target language token)"
            )
        },
    )

    def __post_init__(self):
        pass

@dataclass
class ContinualTrainingArguments(Seq2SeqTrainingArguments):
    cl_method: Optional[str] = field(
        default="MULTI_TASK",
        metadata={"help": "Use the specifid method to do continual learning."}
    )
    order: Optional[int] = field(
        default=1,
        metadata={"help": "task order to perform CL"}
    )
    replay_num_instance_per_task: Optional[int] = field(
        default=None,
        metadata={"help": "The number of instances stored per task, will be used to replay."}
    )
    reg: Optional[float] = field(
        default=None,
        metadata={"help": "Regularization term weights"}
    )
    bottleneck_size: Optional[int] = field(
        default=100,
        metadata={"help": "the bottle neck size for Adapter."}
    )
    number_of_adpt: Optional[int] = field(
        default=40,
        metadata={"help": "number of adapters"}
    )
    early_stopping_patience: Optional[int] = field(
        default=3,
        metadata={"help": "stopping patience"}
    )



def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, ContinualTrainingArguments))
    print(f"len(sys.argv): {len(sys.argv)}")
    print(f"sys.argv: {sys.argv}")
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # logger.info(f"Training/evaluation parameters: {training_args}")
    # logger.info(f"Model parameters: {model_args}")
    # logger.info(f"Data parameters: {data_args}")

    # Sending telemetry. Tracking the example usage helps us better allocate resources to maintain them. The
    # information sent is the one passed as arguments along with your Python/PyTorch versions.
    # send_example_telemetry("run_summarization", model_args, data_args)

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

    if data_args.source_prefix is None and model_args.model_name_or_path in [
        "t5-small",
        "t5-base",
        "t5-large",
        "t5-3b",
        "t5-11b",
    ]:
        logger.warning(
            "You're running a t5 model but didn't provide a source prefix, which is the expected, e.g. with "
            "`--source_prefix 'summarize: ' `"
        )

    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Set seed before initializing model.
    set_seed(training_args.seed)


    logger.info(f"Training/evaluation parameters: {training_args}")
    logger.info(f"Model parameters: {model_args}")
    logger.info(f"Data parameters: {data_args}")

    # get CL task order
    # ordered_cl_task_names = list(train_instances.keys())
    # random.shuffle(ordered_cl_task_names)

    ordered_cl_task_names = []
    if data_args.data_dir_for_task_order is not None:
        with open(f"{data_args.data_dir_for_task_order}/order{training_args.order}.txt") as f:
            ordered_cl_task_names = [line.rstrip() for line in f]

    assert ordered_cl_task_names != [], "Fail to load task orders!"
    logger.info(f"====== ordered_cl_task_names: {ordered_cl_task_names}")

    ################ Load datasets for multi-task training of an initial model ################
    if(training_args.cl_method == "MULTI_TASK"): 
        training_args.multi = True
        training_args.continual = False
    else: 
        training_args.multi = False
        training_args.continual = True
    
    assert training_args.continual == True, "You should use `run_initial_multitask_tuning_with_CL.sh` to run multi-task CL"

    print(f"training_args.multi: {training_args.multi}")
    print(f"training_args.continual: {training_args.continual}")

    ############################ Prepare data for CL training ############################
    # contain all CL tasks and the official test tasks
    raw_datasets = load_dataset(
        # "src/ni_dataset.py", 
        "continual_learning/ni_dataset_for_cl.py", # use modified dadatset script
        data_dir=data_args.data_dir, 
        task_dir=data_args.task_dir, 
        cache_dir=model_args.cache_dir,
        max_num_instances_per_task=data_args.max_num_instances_per_task,
        max_num_instances_per_eval_task=data_args.max_num_instances_per_eval_task,
        task_split_file_name=data_args.task_split_file_name,
        load_official_test=False    # instead we load the official test set below
    )

    print(f"raw_datasets: {raw_datasets}")

    # for each tasks, we need to split train/dev/test; {task_name1: [instance1, ..], ...}
    train_instances, dev_instances, test_instances = train_dev_test_split_by_task(raw_datasets,
        max_num_instances_per_task=data_args.max_num_instances_per_task,
        max_num_instances_per_eval_task=data_args.max_num_instances_per_eval_task,
        continual=training_args.continual
    )


    # load the official and the inital multi-task test data from the disk
    official_test_dataset = load_from_disk(data_args.data_dir_for_official_test)
    multitask_test_dataset = load_from_disk(f"{data_args.data_dir_for_initial_training_dir}/test")
    print(f"official test: {len(official_test_dataset)}")
    print(f"multitask_test_dataset test: {len(multitask_test_dataset)}")
    
    replay_instances = None
    if training_args.cl_method == "REPLAY":
        # also load the data from the inital training set, will be used to jointly train subsequent tasks, 4994
        multitask_train_dataset = load_from_disk(f"{data_args.data_dir_for_initial_training_dir}/train")
        print(f"multitask_train_dataset train: {len(multitask_train_dataset)}")
        replay_instances = get_replay_instances_by_task(multitask_train_dataset, training_args.replay_num_instance_per_task)

    cl_model = Seq2SeqCL(model_args, data_args, training_args)

    # Data collator
    label_pad_token_id = -100 if data_args.ignore_pad_token_for_loss else cl_model.tokenizer.pad_token_id
    data_collator = DataCollatorForNI(
        cl_model.tokenizer,
        model=cl_model.model,
        padding="max_length" if data_args.pad_to_max_length else "longest", # default is "longest" as pad_to_max_length=False
        max_source_length=data_args.max_source_length,
        max_target_length=data_args.max_target_length,
        label_pad_token_id=label_pad_token_id,
        pad_to_multiple_of=8 if training_args.fp16 else None,
        add_task_name=data_args.add_task_name,
        add_task_definition=data_args.add_task_definition,
        num_pos_examples=data_args.num_pos_examples,
        num_neg_examples=data_args.num_neg_examples,
        add_explanation=data_args.add_explanation,
        tk_instruct=data_args.tk_instruct,
        add_task_id=True if training_args.cl_method == "ADAPTERCL" else False # for Adapter CL
    )

    # we don't want to remove unused columns because we will prepare each batch during training, 
    # and some of the information will aslo be used in evaluation.
    training_args.remove_unused_columns = False 

    # output_prediction_folder = f"{training_args.output_dir}/results"
    # if not os.path.isdir(output_prediction_folder):
    #     # if the output directory is not present then create it
    #     os.makedirs(output_prediction_folder)

    # Metric

    def compute_ni_metrics(dataset, preds, save_prefix=None):
        decoded_preds = cl_model.tokenizer.batch_decode(preds, skip_special_tokens=True)
        references = [e["Instance"]["output"] for e in dataset]
        result = compute_metrics(predictions=decoded_preds, references=references)
        result_per_task = compute_grouped_metrics(predictions=decoded_preds, references=references, groups=dataset["Task"])
        result.update(result_per_task)
        categories = ["_".join(it[0].lower().split()) for it in dataset["Categories"]]
        result_per_category = compute_grouped_metrics(predictions=decoded_preds, references=references, groups=categories)
        result.update(result_per_category)
        prediction_lens = [np.count_nonzero(pred != cl_model.tokenizer.pad_token_id) for pred in preds]
        result["gen_len"] = np.mean(prediction_lens)
        result = {k: round(v, 4) for k, v in result.items()}
        if save_prefix is not None:
            with open(os.path.join(output_prediction_folder, f"{save_prefix}_eval_predictions.jsonl"), "w") as fout:
                for example, pred in zip(dataset, decoded_preds):
                    fout.write(json.dumps({
                        "Task": example["Task"],
                        "Definition": example["Definition"],
                        "Instance": example["Instance"],
                        "Prediction": pred
                    }) + "\n")

        # print(f"result: {result}")
        return result

    ############################ CL training ############################
    if training_args.do_train:
        # ordered_cl_task_names = ordered_cl_task_names[:3]
        # ordered_cl_task_names = ['task573_air_dialogue_classification']

        total_cl_tasks = len(ordered_cl_task_names)
        print(f"+++ total CL tasks to run: {total_cl_tasks} ++++")
        for order, task_name in enumerate(ordered_cl_task_names):
            print(f"---------------- order: {order}, task_name: {task_name} ----------------")

            # cl_model.task_list_seen.append(task_name)
            
            # record future task id for Adapter, except for the last task
            if training_args.cl_method == "ADAPTERCL": 
                cl_model.task_list_future.append(task_name)
                # if order < len(ordered_cl_task_names) - 1:
                #     cl_model.task_list_future.append(ordered_cl_task_names[order+1])

            # get current task's train/dev/test set
            task_train_dataset = train_instances[task_name]
            task_dev_dataset = dev_instances[task_name]
            task_test_dataset = test_instances[task_name]
            print(f"task_train_dataset: {len(task_train_dataset)}, task_dev_dataset: {len(task_dev_dataset)}, task_test_dataset: {len(task_test_dataset)}")
            # print(task_dev_dataset[0])

            output_prediction_folder = f"{training_args.output_dir}/results/{order}_{task_name}"
            if not os.path.isdir(output_prediction_folder):
                os.makedirs(output_prediction_folder)

            if training_args.cl_method == "REPLAY":
                # current task train set + task seen replay instances + inital multi-task train set (4994)
                if replay_instances is not None:
                    task_train_dataset = task_train_dataset + cl_model.replay_memory + replay_instances
                else:
                    task_train_dataset = task_train_dataset + cl_model.replay_memory
                print(f"after adding replay memory: {len(cl_model.replay_memory)}, final task_train_dataset: {len(task_train_dataset)}")

            # make them Dataset
            task_train_dataset = Dataset.from_list(task_train_dataset)
            task_dev_dataset = Dataset.from_list(task_dev_dataset)
            task_test_dataset = Dataset.from_list(task_test_dataset)
            
            # Initialize our Trainer
            trainer = NITrainer(
                model=cl_model.model,
                optpar=cl_model.optpar,
                fisher=cl_model.fisher,
                episodic_mem=cl_model.episodic_mem,
                task_list_seen=cl_model.task_list_seen,
                task_list_future=cl_model.task_list_future,
                args=training_args,
                train_dataset=task_train_dataset if training_args.do_train else None,
                eval_dataset=task_dev_dataset if training_args.do_eval else None,
                tokenizer=cl_model.tokenizer,
                data_collator=data_collator,
                compute_metrics=compute_ni_metrics if training_args.predict_with_generate else None, 
                callbacks=[EarlyStoppingCallback(early_stopping_patience=training_args.early_stopping_patience)] if training_args.load_best_model_at_end else None,
            )
            # set the first task to True
            trainer.args.first_task = order == 0

            all_metrics = {"run_name": training_args.run_name}

            checkpoint = None
            if training_args.resume_from_checkpoint is not None:
                checkpoint = training_args.resume_from_checkpoint
            elif last_checkpoint is not None:
                checkpoint = last_checkpoint
            train_result = trainer.train(resume_from_checkpoint=checkpoint)
            trainer.save_model()  # Saves the tokenizer too for easy upload

            metrics = train_result.metrics
            max_train_samples = (
                data_args.max_train_samples if data_args.max_train_samples is not None else len(task_train_dataset)
            )
            metrics["train_samples"] = min(max_train_samples, len(task_train_dataset))

            trainer.log_metrics("train", metrics)
            trainer.save_metrics("train", metrics)
            trainer.save_state()
            all_metrics.update(metrics)

            # Evaluation
            results = {}
            max_length = (
                training_args.generation_max_length
                if training_args.generation_max_length is not None
                else data_args.max_target_length
            )
            num_beams = data_args.num_beams if data_args.num_beams is not None else training_args.generation_num_beams

            # # test on current task's test set
            # if training_args.do_eval:
            #     logger.info("*** Evaluate ***")
            #     metrics = trainer.evaluate(max_length=max_length, num_beams=num_beams, metric_key_prefix="eval")
            #     max_eval_samples = data_args.max_eval_samples if data_args.max_eval_samples is not None else len(task_dev_dataset)
            #     metrics["eval_samples"] = min(max_eval_samples, len(task_dev_dataset))

            #     trainer.log_metrics("eval", metrics)
            #     trainer.save_metrics("eval", metrics)
            #     all_metrics.update(metrics)


            # training_args.do_predict = False

            # test on current task's test set
            if training_args.do_predict:
                logger.info("*** Predict ***")

                ################ predict on current test set ################
                # this is A_{i,i}
                logger.info(f"\n------ Predict on test set of {task_name} ------\n")
                predict_results = trainer.predict(
                    task_test_dataset, metric_key_prefix=f"predict_{order+1}{order+1}", max_length=max_length, num_beams=num_beams
                )
                metrics = predict_results.metrics
                max_predict_samples = (
                    data_args.max_predict_samples if data_args.max_predict_samples is not None else len(task_test_dataset)
                )
                metrics[f"predict_{order+1}{order+1}_samples"] = min(max_predict_samples, len(task_test_dataset))

                trainer.log(metrics)
                trainer.log_metrics(f"predict_{order+1}{order+1}", metrics)
                trainer.save_metrics(f"predict_{order+1}{order+1}", metrics)
                all_metrics.update(metrics)

                # True
                if trainer.is_world_process_zero():
                    if training_args.predict_with_generate:
                        predictions = cl_model.tokenizer.batch_decode(
                            predict_results.predictions, skip_special_tokens=True, clean_up_tokenization_spaces=True
                        )
                        predictions = [pred.strip() for pred in predictions]
                        output_prediction_file = os.path.join(output_prediction_folder, f"predicted_{order+1}{order+1}_examples.jsonl")
                        with open(output_prediction_file, "w") as fout:
                            for example, prediction in zip(task_test_dataset, predictions):
                                example["prediction"] = prediction
                                fout.write(json.dumps(example) + "\n")
                

                ################ predict on the all seen tasks so far individually test set ################
                # used for BWT
                if len(cl_model.task_list_seen) > 0:
                    logger.info(f"\n------ Predict on all task_list_seen={len(cl_model.task_list_seen)} individually test sets ------\n")
                    
                    for task_seen in cl_model.task_list_seen:
                        each_test_set = test_instances[task_seen]

                        each_task_order = ordered_cl_task_names.index(task_seen)
                        each_test_set = Dataset.from_list(each_test_set)
                        print(f"Previous seen task: {task_seen}, order: {each_task_order}, test_set: {len(each_test_set)}")

                        predict_seen_results = trainer.predict(
                            each_test_set, metric_key_prefix=f"predict_seen_{order+1}{each_task_order+1}", max_length=max_length, num_beams=num_beams
                        )
                        metrics = predict_seen_results.metrics
                        max_predict_samples = (
                            data_args.max_predict_samples if data_args.max_predict_samples is not None else len(each_test_set)
                        )
                        metrics[f"predict_seen_{order+1}{each_task_order+1}_samples"] = min(max_predict_samples, len(each_test_set))

                        trainer.log(metrics)
                        trainer.log_metrics(f"predict_seen_{order+1}{each_task_order+1}", metrics)
                        trainer.save_metrics(f"predict_seen_{order+1}{each_task_order+1}", metrics)
                        all_metrics.update(metrics)

                        # True
                        if trainer.is_world_process_zero():
                            if training_args.predict_with_generate:
                                predictions = cl_model.tokenizer.batch_decode(
                                    predict_seen_results.predictions, skip_special_tokens=True, clean_up_tokenization_spaces=True
                                )
                                predictions = [pred.strip() for pred in predictions]
                                output_prediction_file = os.path.join(output_prediction_folder, f"predicted_{order+1}{each_task_order+1}_examples.jsonl")
                                with open(output_prediction_file, "w") as fout:
                                    for example, prediction in zip(each_test_set, predictions):
                                        example["prediction"] = prediction
                                        fout.write(json.dumps(example) + "\n")


                ################ predict on the next task in the sequence test set ################
                # FWT
                if order < len(ordered_cl_task_names) - 1:
                    logger.info(f"\n------ Predict on the next_task={ordered_cl_task_names[order+1]} test sets ------\n")
                    next_task_name = ordered_cl_task_names[order+1]
                    next_task_test_dataset = test_instances[next_task_name]
                    next_task_test_dataset = Dataset.from_list(next_task_test_dataset)
                    print(f"next_task_test_dataset: {len(next_task_test_dataset)}")

                    predict_next_task_results = trainer.predict(
                        next_task_test_dataset, metric_key_prefix=f"predict_next_{order+1}{order+2}", max_length=max_length, num_beams=num_beams
                    )
                    metrics = predict_next_task_results.metrics
                    max_predict_samples = (
                        data_args.max_predict_samples if data_args.max_predict_samples is not None else len(next_task_test_dataset)
                    )
                    metrics[f"predict_next_{order+1}{order+2}_samples"] = min(max_predict_samples, len(next_task_test_dataset))

                    trainer.log(metrics)
                    trainer.log_metrics(f"predict_next_{order+1}{order+2}", metrics)
                    trainer.save_metrics(f"predict_next_{order+1}{order+2}", metrics)
                    all_metrics.update(metrics)

                    # True
                    if trainer.is_world_process_zero():
                        if training_args.predict_with_generate:
                            predictions = cl_model.tokenizer.batch_decode(
                                predict_next_task_results.predictions, skip_special_tokens=True, clean_up_tokenization_spaces=True
                            )
                            predictions = [pred.strip() for pred in predictions]
                            output_prediction_file = os.path.join(output_prediction_folder, f"predicted_next_{order+1}{order+2}_examples.jsonl")
                            with open(output_prediction_file, "w") as fout:
                                for example, prediction in zip(next_task_test_dataset, predictions):
                                    example["prediction"] = prediction
                                    fout.write(json.dumps(example) + "\n")


                # # ONLY predict on the official and init test set after learning the last task
                # # for the long stream
                # if data_args.task_split_file_name == "cl_38_random_tasks" and order < len(ordered_cl_task_names) - 1:
                #     cl_model.task_list_seen.append(task_name)
                    
                #     # save results before skip current task
                #     if (training_args.do_train or training_args.do_eval or training_args.do_predict) and trainer.is_world_process_zero():
                #         with open(os.path.join(output_prediction_folder, "metrics.json"), "w") as fout:
                #             fout.write(json.dumps(all_metrics))
                #     continue

                ################ predict on the official test set ################
                ## DO not predict official and init test for Adapter because it does not have adapters for those tasks
                if (official_test_dataset is not None) and (training_args.cl_method != 'ADAPTERCL') \
                    and not (data_args.task_split_file_name == "cl_38_random_tasks" and order < len(ordered_cl_task_names) - 1) \
                    and not (data_args.task_split_file_name == "cl_dialogue_tasks" and order < len(ordered_cl_task_names) - 1):
                    logger.info(f"\n------ Predict on the official test sets ------\n")
                    predict_official_results = trainer.predict(
                        official_test_dataset, metric_key_prefix="predict_official", max_length=max_length, num_beams=num_beams
                    )
                    metrics = predict_official_results.metrics
                    max_predict_samples = (
                        data_args.max_predict_samples if data_args.max_predict_samples is not None else len(official_test_dataset)
                    )
                    metrics["predict_official_samples"] = min(max_predict_samples, len(official_test_dataset))

                    trainer.log(metrics)
                    trainer.log_metrics("predict_official", metrics)
                    trainer.save_metrics("predict_official", metrics)
                    all_metrics.update(metrics)

                    # True
                    if trainer.is_world_process_zero():
                        if training_args.predict_with_generate:
                            predictions = cl_model.tokenizer.batch_decode(
                                predict_official_results.predictions, skip_special_tokens=True, clean_up_tokenization_spaces=True
                            )
                            predictions = [pred.strip() for pred in predictions]
                            # output_prediction_file = os.path.join(training_args.output_dir, "generated_predictions.txt")
                            # with open(output_prediction_file, "w") as writer:
                            #     writer.write("\n".join(predictions))
                            output_prediction_file = os.path.join(output_prediction_folder, "predicted_official_examples.jsonl")
                            with open(output_prediction_file, "w") as fout:
                                for example, prediction in zip(official_test_dataset, predictions):
                                    example["prediction"] = prediction
                                    fout.write(json.dumps(example) + "\n")


                ################ predict on the initial multi-task test set ################
                if (multitask_test_dataset is not None) and (training_args.cl_method != 'ADAPTERCL') \
                    and not (data_args.task_split_file_name == "cl_38_random_tasks" and order < len(ordered_cl_task_names) - 1)\
                    and not (data_args.task_split_file_name == "cl_dialogue_tasks" and order < len(ordered_cl_task_names) - 1):
                    logger.info(f"\n------ Predict on the initial multi-task test sets ------\n")
                    predict_initial_multi_results = trainer.predict(
                        multitask_test_dataset, metric_key_prefix="predict_initial_multi", max_length=max_length, num_beams=num_beams
                    )
                    metrics = predict_initial_multi_results.metrics
                    max_predict_samples = (
                        data_args.max_predict_samples if data_args.max_predict_samples is not None else len(multitask_test_dataset)
                    )
                    metrics["predict_initial_multi_samples"] = min(max_predict_samples, len(multitask_test_dataset))

                    trainer.log(metrics)
                    trainer.log_metrics("predict_initial_multi", metrics)
                    trainer.save_metrics("predict_initial_multi", metrics)
                    all_metrics.update(metrics)

                    # True
                    if trainer.is_world_process_zero():
                        if training_args.predict_with_generate:
                            predictions = cl_model.tokenizer.batch_decode(
                                predict_initial_multi_results.predictions, skip_special_tokens=True, clean_up_tokenization_spaces=True
                            )
                            predictions = [pred.strip() for pred in predictions]
                            # output_prediction_file = os.path.join(training_args.output_dir, "generated_predictions.txt")
                            # with open(output_prediction_file, "w") as writer:
                            #     writer.write("\n".join(predictions))
                            output_prediction_file = os.path.join(output_prediction_folder, "predicted_initial_multi_examples.jsonl")
                            with open(output_prediction_file, "w") as fout:
                                for example, prediction in zip(multitask_test_dataset, predictions):
                                    example["prediction"] = prediction
                                    fout.write(json.dumps(example) + "\n")


            ####################### after prediction #######################

            # create current task dataloader for EWC, save some training data of this task in the episodic memory
            if training_args.cl_method == 'EWC':
                current_task_train_dataloader = DataLoader(
                    task_train_dataset, shuffle=True, collate_fn=data_collator, batch_size=training_args.per_device_train_batch_size
                )

                for idx_b, b in enumerate(current_task_train_dataloader):
                        cl_model.episodic_mem[task_name].append(b)
                        if idx_b == training_args.replay_num_instance_per_task: break
                # logger.info(f"----------- episodic_mem: {cl_model.episodic_mem} ---------")
                del current_task_train_dataloader

            if training_args.cl_method == 'AGEM':
                current_task_train_dataloader = DataLoader(
                    task_train_dataset, shuffle=True, collate_fn=data_collator, batch_size=training_args.per_device_train_batch_size
                )

                for idx_b, b in enumerate(current_task_train_dataloader):
                        cl_model.episodic_mem["all"].append(b)
                        if idx_b == training_args.replay_num_instance_per_task: break
                del current_task_train_dataloader

            ##### Compute Fisher info Matrix for EWC
            if training_args.cl_method == "EWC" or training_args.cl_method == "L2":
                cl_model.model.cpu()
                for n, p in cl_model.model.named_parameters():
                    cl_model.optpar[n] = torch.Tensor(p.cpu().data)
                    cl_model.fisher[n] = torch.zeros(p.size()) #torch.Tensor(p.cpu().data).zero_()

                if training_args.cl_method == "EWC":
                    for _, batch in enumerate(cl_model.episodic_mem[task_name]):
                        cl_model.model.zero_grad()
                        outputs = cl_model.model(input_ids=batch["input_ids"],
                                                attention_mask=batch["attention_mask"],
                                                labels=batch["labels"])
                        loss = outputs.loss
                        # logger.info(f"----------- loss: {loss} ---------")
                        loss.backward()
                        for n, p in cl_model.model.named_parameters():
                            if p.grad is not None:
                                cl_model.fisher[n].data += p.grad.data ** 2

                        # logger.info(f"----------- cl_model.fisher: {cl_model.fisher} ---------")

                    for name_f,_ in cl_model.fisher.items():
                        cl_model.fisher[name_f] /= len(cl_model.episodic_mem[task_name]) #*hparams.train_batch_size
                    cl_model.model.zero_grad()



            if (training_args.do_train or training_args.do_eval or training_args.do_predict) and trainer.is_world_process_zero():
                with open(os.path.join(output_prediction_folder, "metrics.json"), "w") as fout:
                    fout.write(json.dumps(all_metrics))

            ################################ After training  ################################
            
            if training_args.cl_method == "REPLAY":
                # save some instances of current task to the memory
                available_instances = train_instances[task_name]
                random.shuffle(available_instances)
                selected_instances = available_instances[:training_args.replay_num_instance_per_task]
                cl_model.replay_memory.extend(selected_instances)
                print(f"after the task, save replay_memory: {len(selected_instances)}, current total memory: {len(cl_model.replay_memory)}")
        
            if task_name not in cl_model.task_list_seen:
                cl_model.task_list_seen.append(task_name)

            # delete the trainer obj except for the last task
            if order < total_cl_tasks - 1:
                del trainer

        # # after CL, test on all seen tasks (last row) as the Average Acc.
        # if training_args.do_predict:
        #     logger.info("*** Predict ***")

        #     if len(cl_model.task_list_seen) > 0:
        #         logger.info(f"\n------ Predict on all task_list_seen={len(cl_model.task_list_seen)} test sets ------\n")
                
        #         all_seen_tasks_test_dataset = []
        #         for t in cl_model.task_list_seen:
        #             all_seen_tasks_test_dataset.extend(test_instances[t]) 
        #         all_seen_tasks_test_dataset = Dataset.from_list(all_seen_tasks_test_dataset)
        #         print(f"all_seen_tasks_test_dataset: {len(all_seen_tasks_test_dataset)}")

        #         predict_all_seen_results = trainer.predict(
        #             all_seen_tasks_test_dataset, metric_key_prefix=f"predict_all_seen", max_length=max_length, num_beams=num_beams
        #         )
        #         metrics = predict_all_seen_results.metrics
        #         max_predict_samples = (
        #             data_args.max_predict_samples if data_args.max_predict_samples is not None else len(all_seen_tasks_test_dataset)
        #         )
        #         metrics["predict_all_seen_samples"] = min(max_predict_samples, len(all_seen_tasks_test_dataset))

        #         trainer.log(metrics)
        #         trainer.log_metrics("predict_all_seen", metrics)
        #         trainer.save_metrics("predict_all_seen", metrics)
        #         all_metrics.update(metrics)

        #         # True
        #         if trainer.is_world_process_zero():
        #             if training_args.predict_with_generate:
        #                 predictions = cl_model.tokenizer.batch_decode(
        #                     predict_all_seen_results.predictions, skip_special_tokens=True, clean_up_tokenization_spaces=True
        #                 )
        #                 predictions = [pred.strip() for pred in predictions]
        #                 output_prediction_file = os.path.join(output_prediction_folder, "predicted_all_seen_examples.jsonl")
        #                 with open(output_prediction_file, "w") as fout:
        #                     for example, prediction in zip(all_seen_tasks_test_dataset, predictions):
        #                         example["prediction"] = prediction
        #                         fout.write(json.dumps(example) + "\n")

        # if (training_args.do_train or training_args.do_eval or training_args.do_predict) and trainer.is_world_process_zero():
        #         with open(os.path.join(output_prediction_folder, "metrics.json"), "w") as fout:
        #             fout.write(json.dumps(all_metrics))

            

    return results



def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()