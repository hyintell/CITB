#!/bin/bash
# set -x

# T5
# model="google/t5-xl-lm-adapt"
# model="google/t5-large-lm-adapt"
# model="google/t5-base-lm-adapt"
model="google/t5-small-lm-adapt"

per_device_train_batch_size=8
per_device_eval_batch_size=32

learning_rate=5e-05
num_train_epochs=15
gradient_accumulation_steps=1

# subsequent CL related args, only valid to to train an upper bound model when CL method = "MULTI_TASK"
# for CL method = "Multi", then we also add CL data for multi-task learning
cl_method="MULTI_TASK"
data_dir_for_CL_task="data/splits/CIT_splits" # only used to train an upper bound model
task_split_file_name="cl_38_random_tasks" # the name of the CL task split txt file
stream_name="cl_dialogue_long_tasks"

# for CL tasks
max_num_instances_per_task=100 # we use 100 per task for long stream
max_num_instances_per_eval_task=25 # we use 25 per task for long stream

# data_dir="data/splits/CIT_splits/"
data_dir_for_official_test="data/CIT_data/official_test_data"
data_dir="data/CIT_data/initial_multitask_learning/defintion_pos_2" # the initial multi-task training dataset
task_dir="data/tasks/"


for ((i=1; i<=1; i=i+1))
do 
    seed=$(shuf -i 10-999 -n 1)
    echo Running using seed=$seed

    output_dir="output/continual_instruction_tuning/stream=${stream_name}/CL=${cl_method}/instances${max_num_instances_per_task}_epoch${num_train_epochs}_bs${per_device_train_batch_size}_lr${learning_rate}_seed${seed}"
    run_name="CL=${cl_method}_stream=${stream_name}_seed=${seed}"
    echo $output_dir
    echo $run_name

    python continual_learning/run_initial_multitask_tuning.py \
        --do_train \
        --do_eval \
        --do_predict \
        --predict_with_generate \
        --model_name_or_path $model \
        --data_dir_for_CL_task $data_dir_for_CL_task \
        --task_split_file_name $task_split_file_name \
        --data_dir $data_dir \
        --cl_method $cl_method \
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
        --per_device_train_batch_size $per_device_train_batch_size \
        --per_device_eval_batch_size $per_device_eval_batch_size \
        --gradient_accumulation_steps $gradient_accumulation_steps \
        --learning_rate $learning_rate \
        --num_train_epochs $num_train_epochs \
        --lr_scheduler_type constant \
        --warmup_steps 0 \
        --logging_strategy steps \
        --logging_steps 250 \
        --evaluation_strategy steps \
        --eval_steps 500\
        --save_strategy steps \
        --save_steps 500 \
        --save_total_limit 1\
        --load_best_model_at_end \
        --metric_for_best_model 'rougeL'\
        --bf16 \
        --run_name $run_name \
        --seed $seed
        
    echo
    echo

done