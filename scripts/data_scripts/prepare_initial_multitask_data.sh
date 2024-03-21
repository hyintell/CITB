#!/bin/bash

seed=10
max_num_instances_per_task=100
max_num_instances_per_eval_task=25

data_dir="data/splits/CIT_splits/"
task_dir="data/tasks/"
task_split_file_name="multitask_100_train_tasks"
output_dir="data/CIT_data/initial_multitask_learning"
output_dir_for_official_test="data/CIT_data/official_test_data"


# here --add_task_definition does not affect becuase we just load the data
python continual_learning/prepare_data.py \
    --data_dir $data_dir \
    --task_dir $task_dir \
    --task_split_file_name $task_split_file_name \
    --output_dir $output_dir/defintion_pos_2\
    --save_official_test_tasks True \
    --output_dir_for_official_test $output_dir_for_official_test \
    --max_num_instances_per_task $max_num_instances_per_task \
    --max_num_instances_per_eval_task $max_num_instances_per_eval_task \
    --seed $seed


# # def + 2 pos
# python src/convert_data_to_s2s.py \
#     --data_dir $data_dir \
#     --task_dir $task_dir \
#     --max_num_instances_per_task 100 \
#     --max_num_instances_per_eval_task 100 \
#     --add_task_name False \
#     --add_task_definition True \
#     --num_pos_examples 2 \
#     --num_neg_examples 0 \
#     --add_explanation False \
#     --max_source_length 1024 \
#     --max_target_length 128 \
#     --output_dir $output_dir/defintion_pos_2/
    
