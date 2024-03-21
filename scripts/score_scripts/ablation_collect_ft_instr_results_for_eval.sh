#!/bin/bash


initial_multitask_folder_dir="output/initial_multitask_model/base_epoch15_lr1e-05_seed50"

# results_folder_dir="output/continual_instruction_tuning/stream=cl_dialogue_tasks/CL=FT_INSTR"
# save_metrics_folder="scores/continual_instruction_tuning/stream=cl_dialogue_tasks/CL=FT_INSTR"

# results_folder_dir="output/ablation_continual_instruction_tuning/stream=cl_dialogue_long_tasks/CL=FT_INSTR"
# save_metrics_folder="scores/ablation_continual_instruction_tuning/stream=cl_dialogue_long_tasks/CL=FT_INSTR"

# # ablation: training instances per task
results_folder_dir="output/ablation_continual_instruction_tuning/stream=cl_dialogue_long_tasks/CL=FT_INSTR_n=False_d=True_p2_n2_e=False"
save_metrics_folder="scores/ablation_continual_instruction_tuning/stream=cl_dialogue_long_tasks/CL=FT_INSTR_n=False_d=True_p2_n2_e=False"


# # ablation: training instances per task
# results_folder_dir="output/ablation_continual_instruction_tuning/stream=cl_dialogue_tasks/CL=FT_INSTR_instancesize=10"
# save_metrics_folder="scores/ablation_continual_instruction_tuning/stream=cl_dialogue_tasks/CL=FT_INSTR_instancesize=10"


python collect_results.py \
    --results_folder_dir $results_folder_dir \
    --initial_multitask_folder_dir $initial_multitask_folder_dir \
    --save_metrics_folder $save_metrics_folder
