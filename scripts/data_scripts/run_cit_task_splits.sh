#!/bin/bash

seed=40
cl_task_num=19

# initial_multitask_task_num=100
initial_multitask_task_num=5

output_dir="data/splits/CIT_splits"

python continual_learning/split_tasks_for_cl.py\
    --default_tasks_split_path "data/splits/default"\
    --task_dir "data/tasks"\
    --output_dir $output_dir\
    --track "default"\
    --cl_task_num $cl_task_num\
    --initial_multitask_task_num $initial_multitask_task_num\
    --seed $seed