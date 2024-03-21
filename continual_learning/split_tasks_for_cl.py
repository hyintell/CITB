"""
This script is adapted from https://github.com/allenai/natural-instructions/blob/master/src/split_tasks.py
"""

import sys
import os
import argparse
import glob
import random
import json 
import numpy as np
import pandas as pd


# official task categories for evaluation
test_categories = [
    "Textual Entailment",
    "Cause Effect Classification",
    "Coreference Resolution",
    "Dialogue Act Recognition",
    "Answerability Classification",
    "Overlap Extraction",
    "Word Analogy",
    "Keyword Tagging",
    "Question Rewriting",
    "Title Generation",
    "Data to Text",
    "Grammar Error Correction"
]

# Stream 1: dialogue task categories for subsequent Continual Learning
CL_dialogue_task_categories = [
    # "Dialogue Act Recognition", # this category is in the official test set
    "Intent Identification",
    "Dialogue Generation",
    "Dialogue State Tracking"
]

# Stream 2: randomly select 200 tasks of different types for subsequent Continual Learning


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--default_tasks_split_path', type=str, help='The path to the official split folder')
    parser.add_argument(
        "--task_dir",
        type=str,
        default="tasks/",
        help="the directory path of all the task json files in NaturalInstructions."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="splits/default/",
        help="the directory path for saving the splits text files."
    )
    parser.add_argument(
        "--test_categories",
        nargs="*",
        default=test_categories,
        help="The predefined test categories. Only valid for the cross_category setting."
    )
    parser.add_argument(
        "--cl_task_categories",
        nargs="*",
        default=CL_dialogue_task_categories,
        help="The predefined task categories that are used for subsequent continual learning."
    )
    parser.add_argument(
        "--cl_task_num",
        type=int,
        default=50,
        help="The number of tasks that are used for subsequent continual learning. Tasks will be randomly selected."
            " The remaining tasks will be used to train an initial model. Number should < 700."
    )
    parser.add_argument(
        "--initial_multitask_task_num",
        type=int,
        default=100,
        help="The number of tasks that are used for initial multi-task learning."
    )
    parser.add_argument(
        "--track",
        choices=["default", "xlingual"],
        default="default",
        help="`default` will generate the splits for the English-only track, `xlingual` will generate the splits for the cross-lingual track."
    )
    parser.add_argument(
        "--no_synthetic",
        action="store_true", 
        help="don't include tasks from synthetic source."
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1,
        help="random seed"
    )
    return parser.parse_args()


def official_task_split(args):

    random.seed(args.seed)

    tasks = set()
    task_categories = {}
    task_languages = {}
    task_sources = {}

    for file in glob.glob(os.path.join(args.task_dir, "task*.json")):
        task = os.path.basename(file)[:-5]
        task_info = json.load(open(file, encoding="utf-8"))
        
        if args.no_synthetic:
            source = task_info["Source"]
            if "synthetic" in source:
                continue

        languages = set(task_info["Input_language"] + task_info["Output_language"])
        if not (len(languages) == 1 and languages.pop().lower() == "english"):
            task_languages[task] = "non-en"
        else:
            task_languages[task] = "en"

        task_categories[task] = task_info["Categories"]
        task_sources[task] = task_info["Source"]
        tasks.add(task)

    tasks_by_category = {}
    for task, categories in task_categories.items():
        for category in categories:
            if category not in tasks_by_category:
                tasks_by_category[category] = []
            tasks_by_category[category].append(task)

    tasks_by_source = {}
    for task, sources in task_sources.items():
        for source in sources:
            if source not in tasks_by_source:
                tasks_by_source[source] = []
            tasks_by_source[source].append(task)

    test_categories = set(args.test_categories)
    print(f"Test categories include: {test_categories}")

    train_tasks, test_tasks, excluded_tasks = set(), set(), set()
    
    for category in test_categories:
        if category in tasks_by_category:
            for task in tasks_by_category[category]:
                test_tasks.add(task)
    
    # exclude tasks that have same source as test tasks from the remaining data.
    excluded_tasks = set()
    for test_task in test_tasks:
        for source in task_sources[test_task]:
            for task in tasks_by_source[source]:
                if task in test_tasks:
                    continue
                else:
                    excluded_tasks.add(task)

    # select tasks based on the track, and the other test tasks are excluded from training
    for test_task in test_tasks:
        if args.track == "default" and task_languages[test_task] == "en":
            continue
        elif args.track == "xlingual" and task_languages[test_task] == "non-en":
            continue
        else:
            excluded_tasks.add(test_task)
    
    # moreover, for the default track, we should exclude all the non-en tasks
    if args.track == "default":
        for task in tasks:
            if task_languages[task] == "non-en":
                excluded_tasks.add(task)

    # this task data is not good, need to fix. we remove it from our evaluation temporarily.
    if "task1415_youtube_caption_corrections_grammar_correction" in test_tasks:
        excluded_tasks.add("task1415_youtube_caption_corrections_grammar_correction")

    # make sure the exlucded tasks are not in the test_tasks
    test_tasks = set([t for t in test_tasks if t not in excluded_tasks])

    train_tasks = tasks - excluded_tasks - test_tasks

    # os.makedirs(args.output_dir, exist_ok=True)
    # with open(os.path.join(args.output_dir, "train_tasks.txt"), "w") as fout:
    #     for task in train_tasks:
    #         fout.write(task + "\n")
    # with open(os.path.join(args.output_dir, "test_tasks.txt"), "w") as fout:
    #     for task in test_tasks:
    #         fout.write(task + "\n")
    # with open(os.path.join(args.output_dir, "excluded_tasks.txt"), "w") as fout:
    #     for task in excluded_tasks:
    #         fout.write(task + "\n")


    # check task categories 
    train_task_categories, test_task_categories, exclude_task_categories = set(), set(), set()
    for task in train_tasks:
        if len(task_categories[task]) > 1:
            print(f"This task has multiple category!")
        else: 
            train_task_categories.add(task_categories[task][0])
    for task in test_tasks:
        if len(task_categories[task]) > 1:
            print(f"This task has multiple category!")
        else: 
            test_task_categories.add(task_categories[task][0])
    for task in excluded_tasks:
        if len(task_categories[task]) > 1:
            print(f"This task has multiple category!")
        else: 
            exclude_task_categories.add(task_categories[task][0])

    print(f"\n============ Official Split ============")
    print(f"train_task_categories: {len(train_task_categories)}, train_tasks: {len(train_tasks)}")
    print(f"test_task_categories: {len(test_task_categories)}, test_tasks: {len(test_tasks)}")
    print(f"exclude_task_categories: {len(exclude_task_categories)}, excluded_tasks: {len(excluded_tasks)}")
    """
    train_task_categories: 60, train_tasks: 756
    test_task_categories: 12, test_tasks: 119
    exclude_task_categories: 38, excluded_tasks: 738
    """
    
    # no overlaps
    overlap = train_task_categories.intersection(test_task_categories)
    print(f"train_task_categories and test_task_categories overlap: {len(overlap)}")
    print(f"============ Official Split ============\n")

    return train_tasks, task_categories, tasks_by_category, train_task_categories, test_tasks



if __name__ == "__main__":

    args = parse_args()
    random.seed(args.seed)

    # this is the official split
    train_tasks, task_categories, tasks_by_category, train_task_categories, test_tasks = official_task_split(args)

    if args.cl_task_categories:
        # prepare CL tasks stream by categories, by default using the dialogue stream, which has 3 categories, total 19 tasks
        cl_task_categories = set(args.cl_task_categories)
        print(f"cl_task_categories: {len(cl_task_categories)}, {list(cl_task_categories)[:3]}")

        # prepare CL tasks
        cl_dialogue_tasks = set()
        for category in cl_task_categories:
            if category in tasks_by_category:
                for task in tasks_by_category[category]:
                    cl_dialogue_tasks.add(task)

        diff = cl_dialogue_tasks.difference(train_tasks)
        # these three tasks are excluded by the official split, so we also exclude them 
        # {'task361_spolin_yesand_prompt_response_classification', 'task932_dailydialog_classification', 'task360_spolin_yesand_response_generation'}
        print(f"cl_dialogue_tasks and train_tasks diff: {len(diff)} {diff}")

        # remove tasks that are excluded by the official split
        cl_dialogue_tasks = cl_dialogue_tasks - diff
        # print(f"cl_dialogue_tasks: {len(cl_dialogue_tasks)}")

        # remove cl_dialogue_tasks from train tasks
        train_tasks = train_tasks - cl_dialogue_tasks
        # print(f"train_tasks: {len(train_tasks)}")

        # # remaining train task categories
        # train_task_categories = train_task_categories - cl_task_categories

    if args.cl_task_num:
        print(f"\n** randomly choose {args.cl_task_num} task of various categories for CL **")

        # we should randomly choose args.cl_task_num tasks from the remaining train_tasks + cl_dialogue_tasks
        # it is possible to select dialogue tasks that are in CL_dialogue_task_categories
        # train_tasks.update(cl_dialogue_tasks)
        print(f"train_tasks: {len(train_tasks)}")

        rng = np.random.default_rng(seed=args.seed)
        random_selected_cl_tasks = rng.choice(list(train_tasks), size=args.cl_task_num, replace=False)
        random_selected_cl_tasks = set(random_selected_cl_tasks)
        print(f"random_selected_cl_tasks: {len(random_selected_cl_tasks)}, {list(random_selected_cl_tasks)}")

        # remove random_selected_cl_tasks from train_tasks
        train_tasks = train_tasks - random_selected_cl_tasks - cl_dialogue_tasks

        print(f"train_tasks: {len(train_tasks)}")
    

    """
    We provide two CL task streams:

    Stream 1: 
    dialogue stream, which contain 3 task categories: 
        [
            "Intent Identification",
            "Dialogue Generation",
            "Dialogue State Tracking"
        ],
    in total it has 19 tasks (3 tasks are excluded by the official split)

    Stream 2:
    randomly select additionaly 19 tasks.

    The remaining train tasks are used for multi-task learning to train an initial model.
    For ablation studies, we may use part of the training data.
    """

    # # randomly select multi-task learning tasks from the remaining train tasks set
    rng = np.random.default_rng(seed=args.seed)
    random_select_multitask_learning_tasks = rng.choice(list(train_tasks), size=args.initial_multitask_task_num, replace=False)
    random_select_multitask_learning_tasks = set(random_select_multitask_learning_tasks)

    # check task categories 
    train_task_categories, random_select_multitask_categories, cl_dialogue_task_categories, cl_random_task_categories = set(), set(), set(), set()
    for task in train_tasks:
        if len(task_categories[task]) > 1:
            print(f"This task has multiple category!")
        else: 
            train_task_categories.add(task_categories[task][0])
    for task in random_select_multitask_learning_tasks:
        if len(task_categories[task]) > 1:
            print(f"This task has multiple category!")
        else: 
            random_select_multitask_categories.add(task_categories[task][0])
    for task in cl_dialogue_tasks:
        if len(task_categories[task]) > 1:
            print(f"This task has multiple category!")
        else: 
            cl_dialogue_task_categories.add(task_categories[task][0])
    for task in random_selected_cl_tasks:
        if len(task_categories[task]) > 1:
            print(f"This task has multiple category!")
        else: 
            cl_random_task_categories.add(task_categories[task][0])

    print(f"============ CIT Split ============")
    print(f"remaining train_task_categories: {len(train_task_categories)}, remaining train_tasks: {len(train_tasks)}")
    print(f"random_select_multitask_categories: {len(random_select_multitask_categories)}, random_select_multitask_learning_tasks: {len(random_select_multitask_learning_tasks)}")
    print(f"cl_dialogue_task_categories: {len(cl_dialogue_task_categories)}, cl_dialogue_tasks: {len(cl_dialogue_tasks)}")
    print(f"cl_random_task_categories: {len(cl_random_task_categories)}, random_selected_cl_tasks: {len(random_selected_cl_tasks)}")
    """
    train_task_categories: 54, train_tasks: 542
    cl_dialogue_task_categories: 3, cl_dialogue_tasks: 19
    cl_random_task_categories: 42, random_selected_cl_tasks: 200
    """
    
    # make sure there is no overlap between CL tasks and the tasks used for multi-task instruction tuning
    # but it is possible to have overlap in (random_selected_cl_tasks and cl_dialogue_tasks)
    # (random_select_multitask_learning_tasks and cl_dialogue_tasks), (random_select_multitask_learning_tasks and random_selected_cl_tasks) should have no overlaps
    # so that len(train_tasks) + len(cl_dialogue_tasks) + len(random_selected_cl_tasks) - len(overlap3) = official train_tasks len
    overlap1 = random_select_multitask_learning_tasks.intersection(cl_dialogue_tasks)
    overlap2 = random_select_multitask_learning_tasks.intersection(random_selected_cl_tasks)
    overlap3 = cl_dialogue_tasks.intersection(random_selected_cl_tasks)
    print()
    print(f"random_select_multitask_learning_tasks and cl_dialogue_tasks overlap: {len(overlap1)}")
    print(f"random_select_multitask_learning_tasks and random_selected_cl_tasks overlap: {len(overlap2)}")
    print(f"cl_dialogue_tasks and random_selected_cl_tasks overlap: {len(overlap3)}")

    ## check category overlaps between (random_select_multitask_learning_tasks, random_selected_cl_tasks)
    # their tasks are not overlapping but categories can be overlapped
    overlap4 = random_select_multitask_categories.intersection(cl_random_task_categories)
    print(f"random_select_multitask_categories and cl_random_task_categories overlap: {len(overlap4)}")
    print(overlap4)

    print(f"cate: {random_select_multitask_categories}")
    print(f"random_select_multitask_learning_tasks: {random_select_multitask_learning_tasks}")

    print(f"============ CIT Split ============")

    # # cl_task_splits_folder = f"{args.output_dir}"
    # # os.makedirs(cl_task_splits_folder, exist_ok=True)

    # # with open(os.path.join(cl_task_splits_folder, f"multitask_{args.initial_multitask_task_num}_train_tasks.txt"), "w") as fout:
    # #     for task in random_select_multitask_learning_tasks:
    # #         fout.write(task + "\n")
    # # with open(os.path.join(cl_task_splits_folder, "cl_dialogue_tasks.txt"), "w") as fout:
    # #     for task in cl_dialogue_tasks:
    # #         fout.write(task + "\n")
    # with open(os.path.join(cl_task_splits_folder, f"cl_{args.cl_task_num}_random_tasks.txt"), "w") as fout:
    #     for task in random_selected_cl_tasks:
    #         fout.write(task + "\n")

    # ## aslo save the additional args.cl_task_num tasks + the cl_dialogue_tasks tasks together
    # with open(os.path.join(cl_task_splits_folder, f"cl_{args.cl_task_num+len(cl_dialogue_tasks)}_random_tasks.txt"), "w") as fout:
    #     total_ = random_selected_cl_tasks | cl_dialogue_tasks
    #     print(f"total_: {len(total_)}")
    #     for task in total_:
    #         fout.write(task + "\n")
    
    # # # also save the official test_tasks 
    # # with open(os.path.join(cl_task_splits_folder, "test_tasks.txt"), "w") as fout:
    # #     for task in test_tasks:
    # #         fout.write(task + "\n")

    # # also save the task categories 
    # # with open(os.path.join(cl_task_splits_folder, f"multitask_{args.initial_multitask_task_num}_train_task_categories.txt"), "w") as fout:
    # #     for task in random_select_multitask_categories:
    # #         fout.write(task + "\n")
    # with open(os.path.join(cl_task_splits_folder, f"cl_{args.cl_task_num}_random_task_categories.txt"), "w") as fout:
    #     for task in cl_random_task_categories:
    #         fout.write(task + "\n")
    # # with open(os.path.join(cl_task_splits_folder, f"cl_dialogue_task_categories.txt"), "w") as fout:
    # #     for task in cl_dialogue_task_categories:
    # #         fout.write(task + "\n")