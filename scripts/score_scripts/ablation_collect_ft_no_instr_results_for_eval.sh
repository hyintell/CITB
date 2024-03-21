#!/bin/bash


# results_folder_dir="output/continual_instruction_tuning/stream=cl_dialogue_tasks/CL=FT_NO_INSTR"
# save_metrics_folder="scores/continual_instruction_tuning/stream=cl_dialogue_tasks/CL=FT_NO_INSTR"

# results_folder_dir="output/ablation_continual_instruction_tuning/stream=cl_dialogue_long_tasks/CL=FT_NO_INSTR_LM"
# save_metrics_folder="scores/ablation_continual_instruction_tuning/stream=cl_dialogue_long_tasks/CL=FT_NO_INSTR_LM"

# # task name only
# results_folder_dir="output/ablation_continual_instruction_tuning/stream=cl_dialogue_long_tasks/CL=FT_NO_INSTR_LM_n=True_d=False_p0_n0_e=0"
# save_metrics_folder="scores/ablation_continual_instruction_tuning/stream=cl_dialogue_long_tasks/CL=FT_NO_INSTR_LM_n=True_d=False_p0_n0_e=0"

# # def only
results_folder_dir="output/ablation_continual_instruction_tuning/stream=cl_dialogue_long_tasks/CL=FT_NO_INSTR_LM_n=False_d=True_p2_n2_e=False"
save_metrics_folder="scores/ablation_continual_instruction_tuning/stream=cl_dialogue_long_tasks/CL=FT_NO_INSTR_LM_n=False_d=True_p2_n2_e=False"

# ablation: training instances per task
# results_folder_dir="output/ablation_continual_instruction_tuning/stream=cl_dialogue_tasks/CL=FT_NO_INSTR_LM_instancesize=200"
# save_metrics_folder="scores/ablation_continual_instruction_tuning/stream=cl_dialogue_tasks/CL=FT_NO_INSTR_LM_instancesize=200"


python collect_results.py \
    --results_folder_dir $results_folder_dir \
    --save_metrics_folder $save_metrics_folder
