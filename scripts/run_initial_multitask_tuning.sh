#!/bin/bash
# set -x

# T5
# model="google/t5-xl-lm-adapt"
# model="google/t5-large-lm-adapt"
# model="google/t5-base-lm-adapt"
model="google/t5-small-lm-adapt"

per_device_train_batch_size=8
per_device_eval_batch_size=32

seed=$(shuf -i 10-999 -n 1)
learning_rate=1e-05
num_train_epochs=15
gradient_accumulation_steps=1

data_dir_for_official_test="data/CIT_data/official_test_data"
data_dir="data/CIT_data/initial_multitask_learning/defintion_pos_2" 
task_dir="data/tasks/"
output_dir="output/initial_multitask_model/base_epoch${num_train_epochs}_lr${learning_rate}_seed${seed}"

python continual_learning/run_initial_multitask_tuning.py \
    --do_train \
    --do_eval \
    --do_predict \
    --predict_with_generate \
    --model_name_or_path $model \
    --data_dir $data_dir \
    --data_dir_for_official_test $data_dir_for_official_test \
    --max_source_length 1024 \
    --max_target_length 128 \
    --generation_max_length 128 \
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
    --run_name "initial_multitask_model" \
    --seed $seed
