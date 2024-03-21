#!/bin/bash


# results_folder_dir="output/continual_instruction_tuning/stream=cl_dialogue_long_tasks/CL=FT_NO_INSTR_LM"
# save_metrics_folder="scores/continual_instruction_tuning/stream=cl_dialogue_long_tasks/CL=t_FT_NO_INSTR_LM"

# results_folder_dir="output/continual_instruction_tuning/stream=cl_dialogue_tasks/CL=FT_NO_INSTR"
# save_metrics_folder="scores/continual_instruction_tuning/stream=cl_dialogue_tasks/CL=FT_NO_INSTR"


python collect_results.py \
    --results_folder_dir $results_folder_dir \
    --save_metrics_folder $save_metrics_folder
