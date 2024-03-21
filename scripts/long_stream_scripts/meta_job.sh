#!/bin/bash

echo "++++++++++++++++++++++++++ Runing run_cit_ft_instr.sh ++++++++++++++++++++++++++"
bash ./scripts/long_stream_scripts/run_cit_ft_instr.sh
echo

echo "++++++++++++++++++++++++++ Runing run_cit_ft_no_instr.sh ++++++++++++++++++++++++++"
bash ./scripts/long_stream_scripts/run_cit_ft_no_instr.sh
echo

echo "++++++++++++++++++++++++++ Runing run_cit_l2.sh ++++++++++++++++++++++++++"
bash ./scripts/long_stream_scripts/run_cit_l2.sh
echo

echo "++++++++++++++++++++++++++ Runing run_cit_ewc.sh ++++++++++++++++++++++++++"
bash ./scripts/long_stream_scripts/run_cit_ewc.sh
echo

echo "++++++++++++++++++++++++++ Runing run_cit_agem_10.sh ++++++++++++++++++++++++++"
bash ./scripts/long_stream_scripts/run_cit_agem_10.sh
echo

echo "++++++++++++++++++++++++++ Runing run_cit_replay_10.sh ++++++++++++++++++++++++++"
bash ./scripts/long_stream_scripts/run_cit_replay_10.sh
echo

echo "++++++++++++++++++++++++++ Runing run_initial_multitask_tuning_with_CL.sh ++++++++++++++++++++++++++"
bash ./scripts/long_stream_scripts/run_initial_multitask_tuning_with_CL.sh
echo

echo "++++++++++++++++++++++++++ Runing run_cit_ewc.sh ++++++++++++++++++++++++++"
bash ./scripts/long_stream_scripts/run_cit_ewc.sh
echo

echo "++++++++++++++++++++++++++ Runing run_cit_adaptercl.sh ++++++++++++++++++++++++++"
bash ./scripts/long_stream_scripts/run_cit_adaptercl.sh
echo

echo "++++++++++++++++++++++++++ End ++++++++++++++++++++++++++"

