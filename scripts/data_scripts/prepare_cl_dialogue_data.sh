#!/bin/bash

seed=10
# In CL, for each task, the maximum training instances are 500
max_num_instances_per_task=500 
max_num_instances_per_eval_task=50

data_dir="data/splits/CIT_splits/"
task_dir="data/tasks/"
task_split_file_name="cl_dialogue_tasks"
output_dir="data/CIT_data/cl_dialogue_tasks"


python continual_learning/prepare_data.py \
    --data_dir $data_dir \
    --task_dir $task_dir \
    --task_split_file_name $task_split_file_name \
    --output_dir $output_dir/defintion_pos_2\
    --max_num_instances_per_task $max_num_instances_per_task \
    --max_num_instances_per_eval_task $max_num_instances_per_eval_task \
    --seed $seed
    
