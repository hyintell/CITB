#!/bin/bash
# set -x

# T5
# model="google/t5-xl-lm-adapt"
# model="google/t5-large-lm-adapt"
# model="google/t5-base-lm-adapt"
# model="google/t5-small-lm-adapt"
model="output/initial_multitask_model/base_epoch15_lr1e-05_seed50/checkpoint-14000"

per_device_eval_batch_size=32

cl_method="EVAL_INIT"
data_dir_for_CL_task="data/splits/CIT_splits" # only used to train an upper bound model
task_split_file_name="cl_dialogue_tasks" # the name of the CL task split txt file

max_num_instances_per_task=500 
max_num_instances_per_eval_task=50

# data_dir="data/splits/CIT_splits/"
data_dir_for_official_test="data/CIT_data/official_test_data"
data_dir="data/CIT_data/initial_multitask_learning/defintion_pos_2" 
task_dir="data/tasks/"


for ((i=1; i<=2; i=i+1))
do 
    seed=$(shuf -i 10-999 -n 1)
    echo "Running using seed=$seed"

    output_dir="output/initial_multitask_model/eval_base_epoch${num_train_epochs}_lr${learning_rate}_seed${seed}"
    run_name="CL=${cl_method}_stream=${task_split_file_name}_seed=${seed}"
    echo $output_dir
    echo $run_name

    python continual_learning/run_initial_multitask_tuning.py \
        --do_predict \
        --predict_with_generate \
        --model_name_or_path $model \
        --data_dir $data_dir \
        --cl_method $cl_method \
        --data_dir_for_CL_task $data_dir_for_CL_task \
        --task_split_file_name $task_split_file_name \
        --data_dir_for_official_test $data_dir_for_official_test \
        --max_source_length 1024 \
        --max_target_length 128 \
        --generation_max_length 128 \
        --max_num_instances_per_task $max_num_instances_per_task \
        --max_num_instances_per_eval_task $max_num_instances_per_eval_task \
        --add_task_name False \
        --add_task_definition True \
        --num_pos_examples 2 \
        --num_neg_examples 0 \
        --add_explanation False \
        --tk_instruct False \
        --data_dir $data_dir \
        --task_dir $task_dir \
        --output_dir $output_dir \
        --overwrite_output_dir \
        --cache_dir ./cache/ \
        --overwrite_cache \
        --per_device_eval_batch_size $per_device_eval_batch_size \
        --run_name $run_name \
        --seed $seed
            
    echo
    echo

done
