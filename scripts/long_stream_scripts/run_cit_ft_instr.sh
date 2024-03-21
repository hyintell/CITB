#!/bin/bash
# set -x


# training args
# model="t5-small"
model="output/initial_multitask_model/base_epoch15_lr1e-05_seed50/checkpoint-14000"
# seed=$(shuf -i 10-999 -n 1)
learning_rate=1e-05
num_train_epochs=15
per_device_train_batch_size=16
per_device_eval_batch_size=32
gradient_accumulation_steps=1

max_num_instances_per_task=100 # we use 100 per task for long stream
max_num_instances_per_eval_task=25 # we use 25 per task for long stream

# CL args
cl_method="FT_INSTR"

# other args
data_dir="data/splits/CIT_splits/"
task_dir="data/tasks/"
data_dir_for_official_test="data/CIT_data/official_test_data"
data_dir_for_initial_training_dir="data/CIT_data/initial_multitask_learning/defintion_pos_2" 
task_split_file_name="cl_38_random_tasks" # the name of the CL task split txt file

stream_name="cl_dialogue_long_tasks"
data_dir_for_task_order="data/CIT_data/task_orders/stream=${stream_name}"


# for order in 1 2 3
for order in 1
do 
    seed=$(shuf -i 10-999 -n 1)
    echo Running using seed=$seed

    output_dir="output/continual_instruction_tuning/stream=${stream_name}/CL=${cl_method}/epoch${num_train_epochs}_bs${per_device_train_batch_size}_lr${learning_rate}_order${order}_seed${seed}"
    run_name="CL=${cl_method}_stream=${stream_name}_seed=${seed}"
    echo $output_dir
    echo $run_name

    python continual_learning/run_continual_instruct_tuning.py\
        --do_train \
        --do_eval \
        --do_predict \
        --predict_with_generate \
        --model_name_or_path $model \
        --cl_method $cl_method \
        --order $order \
        --data_dir_for_task_order $data_dir_for_task_order \
        --task_split_file_name $task_split_file_name \
        --data_dir_for_official_test $data_dir_for_official_test \
        --data_dir_for_initial_training_dir $data_dir_for_initial_training_dir \
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
        --logging_steps 50 \
        --evaluation_strategy epoch \
        --save_strategy epoch \
        --save_total_limit 1\
        --load_best_model_at_end \
        --metric_for_best_model 'rougeL'\
        --bf16 \
        --run_name $run_name \
        --seed $seed
    
    echo
    echo

done



    
