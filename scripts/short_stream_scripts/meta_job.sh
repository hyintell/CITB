#!/bin/bash

echo "++++++++++++++++++++++++++ Runing run_cit_adaptercl.sh ++++++++++++++++++++++++++"
bash ./scripts/run_cit_adaptercl.sh
echo

echo "++++++++++++++++++++++++++ Runing run_cit_adaptercl_200.sh ++++++++++++++++++++++++++"
bash ./scripts/run_cit_adaptercl_200.sh
echo

echo "++++++++++++++++++++++++++ Runing run_cit_ft_instr.sh ++++++++++++++++++++++++++"
bash ./scripts/run_cit_ft_instr.sh
echo

echo "++++++++++++++++++++++++++ Runing run_cit_l2.sh ++++++++++++++++++++++++++"
bash ./scripts/run_cit_l2.sh
echo

echo "++++++++++++++++++++++++++ Runing run_cit_replay_10.sh ++++++++++++++++++++++++++"
bash ./scripts/run_cit_replay_10.sh
echo

echo "++++++++++++++++++++++++++ Runing run_cit_ewc.sh ++++++++++++++++++++++++++"
bash ./scripts/run_cit_ewc.sh
echo

echo "++++++++++++++++++++++++++ Runing run_cit_replay.sh ++++++++++++++++++++++++++"
bash ./scripts/run_cit_replay.sh
echo

echo "++++++++++++++++++++++++++ Runing run_cit_agem_10.sh ++++++++++++++++++++++++++"
bash ./scripts/run_cit_agem_10.sh
echo

echo "++++++++++++++++++++++++++ End ++++++++++++++++++++++++++"

