#!/bin/bash


results_folder_dir="output/continual_instruction_tuning/stream=cl_dialogue_long_tasks/CL=MULTI_TASK"
save_metrics_folder="scores/continual_instruction_tuning/stream=cl_dialogue_long_tasks/CL=MULTI"

python collect_results.py \
    --results_folder_dir $results_folder_dir \
    --save_metrics_folder $save_metrics_folder