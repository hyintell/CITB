
import re
import sys
import os
import json
import numpy as np
import argparse
import glob
import itertools
from collections import defaultdict
from pprint import pprint
from tabulate import tabulate


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--initial_multitask_folder_dir", type=str, default="", help="Path to the initial multi-task trained model folde")
    parser.add_argument("--results_folder_dir", type=str, default="", help="Path to the folder with the results")
    parser.add_argument("--save_metrics_folder", type=str, default="", help="Path to save the scores")

    args = parser.parse_args()

    return args


def collect_cl_metrics_by_each_task(results_json, current_task_order, total_task_num, metric="rougeL"):

    # get the score of current task, i.e., a_{i, i}
    current_task_score = results_json[f"predict_{current_task_order}{current_task_order}_{metric}"]

    next_task_score = None
    # except for the last task, get the score of the next task, i.e., a_{i, i+1}
    if current_task_order < total_task_num:
        next_task_score = results_json[f"predict_next_{current_task_order}{current_task_order+1}_{metric}"]
    
    all_seen_tasks_scores = []
    # except for the first task, get all the scores of seen tasks, i.e., a_{i, [1:i-1]}
    if current_task_order > 1:
        for i in list(range(1, current_task_order)):
            previous_task_score = results_json[f"predict_seen_{current_task_order}{i}_{metric}"]
            all_seen_tasks_scores.append(previous_task_score)

    # get the score of the official test tasks
    if f"predict_official_{metric}" in results_json:
        official_test_score = results_json[f"predict_official_{metric}"]
    else: 
        official_test_score = 0

    # get the score of the initial multi-task test tasks
    if f"predict_initial_multi_{metric}" in results_json: 
        initial_multi_test_score = results_json[f"predict_initial_multi_{metric}"]
    else: 
        initial_multi_test_score = 0

    ################## get task category
    
    # get training metrics
    train_runtime = results_json[f"train_runtime"]
    train_epoch = results_json[f"epoch"]
    train_samples = results_json[f"train_samples"]

    metrics = {
        'current_task_score': current_task_score,
        'next_task_score': next_task_score,
        'all_seen_tasks_scores': all_seen_tasks_scores,
        'official_test_score': official_test_score,
        'initial_multi_test_score': initial_multi_test_score,
        'train_runtime': train_runtime,
        'train_epoch': train_epoch,
        'train_samples': train_samples
    }

    return metrics


def collect_cl_metrics_by_seed(path, metric='rougeL'):
    """ For each seed experiments, 
        read through each CL task folder by task order and collect all metrics 
    """

    cl_task_folders = glob.glob(f"{path}/*")
    
    # get ordered CL tasks
    ordered_cl_tasks = []
    for folder in cl_task_folders:
        task_ = folder.split('results/')[1]

        # get task order and task_name
        task_order, task_name = task_.split('_', 1)
        task_order = int(task_order)
        # print(f"task_order: {task_order}, task_name: {task_name}")
        ordered_cl_tasks.append((task_order, task_name))

    # sort by task order
    ordered_cl_tasks = sorted(ordered_cl_tasks, key=lambda x: x[0])
    total_task_num = len(ordered_cl_tasks)

    results_matrix = []
    # collect metrics for each CL task
    for (task_order, task_name) in ordered_cl_tasks:
        current_task_path = [p for p in cl_task_folders if task_name in p][0]

        results_json = json.load(open(f"{current_task_path}/metrics.json"))
        # here use task_order+1 because we record the order from 1, not from 0, in the results.json
        metrics = collect_cl_metrics_by_each_task(results_json, task_order+1, total_task_num, metric=metric)

        # create a TxT matrix
        # build a task row, each row should have the same length, which is total_task_num
        # each row is made of all seen tasks + the task itself + next task
        if metrics['next_task_score'] is None:
            # reach the last task, which does not have the next task prediction
            each_row = metrics['all_seen_tasks_scores'] + [metrics['current_task_score']]
        else:
            each_row = metrics['all_seen_tasks_scores'] + [metrics['current_task_score']] + [metrics['next_task_score']]

        # pad each row to the same length of total_task_num
        if len(each_row) < total_task_num:
            pad_num = total_task_num - len(each_row)
            each_row.extend(['-'] * pad_num)

        # also append official test results and initial multi-task test results, and training details
        each_row = each_row + [metrics['official_test_score']] + [metrics['initial_multi_test_score']] + [metrics['train_runtime']] \
            + [metrics['train_epoch']] + [metrics['train_samples']]

        results_matrix.append(each_row)

    return results_matrix, ordered_cl_tasks, total_task_num


def nan_if(arr, value):
    return np.where(arr == value, np.nan, arr)


def print_table(results_matrix, total_task_num, tablefmt='simple', write2file=False, save_path=None):
    headers = [f"T{i+1}" for i in list(range(total_task_num))]
    headers = headers + ['official_test', 'initial_multi', 'train_runtime', 'train_epoch', 'train_samples']
    rows = [f"T{i+1}" for i in list(range(total_task_num))]

    if write2file:
        print(tabulate(results_matrix, headers=headers, showindex=rows, numalign='center', tablefmt=tablefmt), file=open(save_path, 'w'))
    else:
        print(tabulate(results_matrix, headers=headers, showindex=rows, numalign='center', tablefmt=tablefmt))



def compute_final_average_acc(results_matrix, total_task_num):
    """
        A_T = 1/T * [a_{T,i}]
        After learning the final task, get the average results of all tasks
    """
    # last row
    last_task_row = results_matrix[-1]
    # get all scores until total_task_num
    all_scores = last_task_row[:total_task_num]
    average_accuracy = np.array(all_scores).mean()
    # average_accuracy = round(average_accuracy, 1)
    
    return average_accuracy


def compute_average_all_seen_acc(results_matrix, total_task_num):
    """
        After learning the each task, get the average results of all seen tasks
    """

    all_seen_task_scores = []
    for current_task_order, each_row in enumerate(results_matrix):
        # get the average score of each row -> all seen tasks
        all_seen_task = each_row[:current_task_order+1]
        all_seen_task_avg_score = np.array(all_seen_task).mean()
        # all_seen_task_avg_score = round(np.array(all_seen_task).mean(), 4)
        all_seen_task_scores.append(all_seen_task_avg_score)
    
    return all_seen_task_scores


def compute_FWT(results_matrix, total_task_num):
    """
        FWT_T = 1/(T-1) * [a_{i-1,i}}]
        After learning the current task, get the prediction of the next task in the future
    """

    all_next_task_scores = []
    for current_task_order, each_row in enumerate(results_matrix):
        # get next score
        if current_task_order < total_task_num - 1:
            next_task_score = each_row[current_task_order+1]
            all_next_task_scores.append(next_task_score)

    average_FWT = np.array(all_next_task_scores).mean()
    # average_FWT = round(average_FWT, 1)

    return average_FWT


def compute_BWT(results_matrix, total_task_num):
    """
        BWT_T = 1/(T-1) * [( a_{T,i} - a_{i,i} )]
        After learning the final task, get the subtraction of the current prediction and the previous prediction
    """

    all_diagonal_scores = []
    for current_task_order, each_row in enumerate(results_matrix):
        # except for the last one
        if current_task_order < total_task_num - 1:
            all_diagonal_scores.append(each_row[current_task_order])


    # last row
    last_task_row = results_matrix[-1]
    # get all scores until total_task_num-1
    all_last_scores = last_task_row[:total_task_num-1]

    average_BWT = (np.array(all_last_scores) - np.array(all_diagonal_scores)).mean()
    # average_BWT = round(average_BWT, 1)

    return average_BWT


def calculate_mean_std(all_metric_scores):
    """ Calculate mean and std across different seeds """

    mean_results = {}
    std_results = {}
    for key, value in all_metric_scores.items():
        # average across the lists, if the value is a list of lists
        if isinstance(value[0], list):
            input_lists = [np.array(v) for v in value]
            mean_results[key] = [round(np.mean(k), 1) for k in zip(*input_lists)]
            std_results[key] = [round(np.std(k), 1) for k in zip(*input_lists)]
        else:
            mean_results[key] = round( np.array(value).mean(), 1)
            std_results[key] = round(np.array(value).std(), 1)

    return mean_results, std_results


def merge_seed_results(seed_results):
    """ merge the results from different seeds together, append as a list """

    merged_results_across_seeds = defaultdict(list)

    for seed, per_seed_result in seed_results.items():        
        for key, value in per_seed_result.items():
            merged_results_across_seeds[key].extend(value)


    return merged_results_across_seeds


def prepare_one_seed_result_for_CL(path, metric):

    results_matrix, ordered_cl_tasks, total_task_num = collect_cl_metrics_by_seed(path, metric)

    # collect_multi_and_base_metrics_by_seed(path, metric)

    average_accuracy = compute_final_average_acc(results_matrix, total_task_num)
    average_FWT = compute_FWT(results_matrix, total_task_num)
    average_BWT = compute_BWT(results_matrix, total_task_num)
    all_seen_task_scores = compute_average_all_seen_acc(results_matrix, total_task_num)
    
    # get list of 'official_test_score','initial_multi_test_score','train_runtime','train_epoch','train_samples when learning
    official_test_score = [row[total_task_num] for row in results_matrix]
    initial_multi_test_score = [row[total_task_num+1] for row in results_matrix]
    average_train_runtime = np.array([row[total_task_num+2] for row in results_matrix]).mean()
    total_train_runtime = np.array([row[total_task_num+2] for row in results_matrix]).sum()
    average_train_epoch = np.array([row[total_task_num+3] for row in results_matrix]).mean()
    average_train_samples = np.array([row[total_task_num+4] for row in results_matrix]).mean()

    final_official_test_score = official_test_score[-1]
    final_initial_multi_test_score = initial_multi_test_score[-1]

    print(f"\n************ Metrics = {metric} ************")
    print(f"average_accuracy: {average_accuracy}")
    print(f"average_FWT: {average_FWT}")
    print(f"average_BWT: {average_BWT}")
    print(f"final_official_test_score: {final_official_test_score}")
    print(f"final_initial_multi_test_score: {final_initial_multi_test_score}")
    print(f"average_train_runtime: {average_train_runtime}")
    print(f"total_train_runtime: {total_train_runtime}")
    print(f"average_train_epoch: {average_train_epoch}")
    print(f"average_train_samples: {average_train_samples}")

    # lists
    print(f"all_seen_task_scores: {all_seen_task_scores}")
    print(f"official_test_score: {official_test_score}")
    print(f"initial_multi_test_score: {initial_multi_test_score}")

    # collect all scores for average the results
    per_seed_metric_scores = defaultdict(list)
    per_seed_metric_scores["average_accuracy"].append(average_accuracy) 
    per_seed_metric_scores["average_FWT"].append(average_FWT)
    per_seed_metric_scores["average_BWT"].append(average_BWT)
    per_seed_metric_scores["final_initial_multi_test_score"].append(final_initial_multi_test_score)
    per_seed_metric_scores["final_official_test_score"].append(final_official_test_score)

    per_seed_metric_scores["total_train_runtime"].append(total_train_runtime)
    per_seed_metric_scores["average_train_runtime"].append(average_train_runtime)
    per_seed_metric_scores["average_train_epoch"].append(average_train_epoch)
    per_seed_metric_scores["average_train_samples"].append(average_train_samples)
    
    per_seed_metric_scores["all_seen_task_scores"].append(all_seen_task_scores)
    per_seed_metric_scores["initial_multi_test_score"].append(initial_multi_test_score)
    per_seed_metric_scores["official_test_score"].append(official_test_score)
    
    return results_matrix, ordered_cl_tasks, total_task_num, per_seed_metric_scores


def prepare_one_seed_result_for_multi(path, metric):

    metrics_path = glob.glob(f"{path}/*.json")[0]
    # print(metrics_path)

    results_json = json.load(open(metrics_path))

    # get the score of the official test tasks
    official_test_score = results_json[f"predict_official_{metric}"]

    # get the score of the initial multi-task test tasks
    initial_multi_test_score = results_json[f"predict_{metric}"]

    # get the score of the CL test tasks
    cl_test_score = results_json[f"predict_cl_{metric}"]
    # print(f"cl_test_score: {cl_test_score}")

    ################## get task category
    
    # get training metrics
    train_runtime = results_json[f"train_runtime"]
    train_epoch = results_json[f"epoch"]
    train_samples = results_json[f"train_samples"]

    metrics = {
        'official_test_score': official_test_score,
        'initial_multi_test_score': initial_multi_test_score,
        'cl_test_score': cl_test_score,
        'train_runtime': train_runtime,
        'train_epoch': train_epoch,
        'train_samples': train_samples
    }

    return metrics


def collect_cl_results(args, cl_stream, cl_method):
    """ collect all results for a CL method """

    # for each seed in the experiments
    seed_experiments = glob.glob(f"{args.results_folder_dir}/*")

    print()
    
    all_experiment_orders = set()
    # get all task orders
    for seed_folder in seed_experiments:
        order = seed_folder.split('order')[1].split('_')[0]
        order = int(order)
        all_experiment_orders.add(order)
    print(f"all_experiment_orders: {all_experiment_orders}")

    print(seed_experiments)


    all_metric_scores = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
    all_task_orders = defaultdict(dict)
    all_results_matrix = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))

    all_task_orders_latex_table = []
    # for the same task order, get the averaged results
    for experiment_order in list(all_experiment_orders):
        
        for seed_folder in seed_experiments:
            if f"order{experiment_order}_" not in seed_folder:
                continue

            # get seed
            seed = seed_folder.split('seed')[1]
            seed = int(seed)
            print(f"========================== Collecting results for {cl_stream}, experiment_order: {experiment_order} seed: {seed} ==========================")
            
            if cl_method == 'EWC' and seed == 40:
                continue 

            # if cl_method == 'FT_NO_INSTR_LM' and seed == 302:
            #     continue 

            for metric in ['rougeL', 'rouge1', 'exact_match']:
                results_matrix, ordered_cl_tasks, total_task_num, per_seed_metric_scores = prepare_one_seed_result_for_CL(
                    f"{seed_folder}/results/", metric=metric)

                # print(f"per_seed_metric_scores: {per_seed_metric_scores}")

                # print_table(results_matrix, total_task_num, tablefmt='simple', write2file=False, save_path=None)

                print_table(results_matrix, total_task_num, tablefmt='simple', write2file=False, save_path=None)
                print_table(results_matrix, total_task_num, tablefmt='simple', write2file=True, 
                    save_path=f"{args.save_metrics_folder}/table_order={experiment_order}_metric={metric}_seed={seed}.txt")
                print_table(results_matrix, total_task_num, tablefmt='latex_booktabs', write2file=True, 
                    save_path=f"{args.save_metrics_folder}/table_order={experiment_order}_metric={metric}_seed={seed}_latex.txt")
                # print_table(results_matrix, total_task_num, tablefmt='latex_booktabs', write2file=False, save_path=None)
                print()

                all_metric_scores[f"{experiment_order}"][metric][f"seed={seed}"] = per_seed_metric_scores
                all_results_matrix[f"exp_order={experiment_order}"][metric][f"seed={seed}"] = results_matrix
            # all_results_matrix[f"seed={seed}"] = results_matrix

            task_orders = [t[1] for t in ordered_cl_tasks]
            all_task_orders[f"{experiment_order}"][f"seed={seed}"] = task_orders

        # # save task orders to latex
        # headers = ["Task Order", "Task"]
        # task_num = len(task_orders)
        # task_order_per_order = [[experiment_order, task] for task in task_orders]
        # all_task_orders_latex_table.extend(task_order_per_order)
        # print(task_order_per_order)

        # print(tabulate(task_order_per_order, headers=headers, numalign='center', tablefmt="latex_booktabs"))

        # sys.exit(0)

        # headers = [f"T{i+1}" for i in list(range(total_task_num))]
        # headers = headers + ['official_test', 'initial_multi', 'train_runtime', 'train_epoch', 'train_samples']
        # rows = [f"T{i+1}" for i in list(range(total_task_num))]

        # if write2file:
        #     print(tabulate(results_matrix, headers=headers, showindex=rows, numalign='center', tablefmt=tablefmt), file=open(save_path, 'w'))
        # else:
        #     print(tabulate(results_matrix, headers=headers, showindex=rows, numalign='center', tablefmt=tablefmt))


        # print_table(results_matrix, total_task_num, tablefmt='simple', write2file=True, 
        #             save_path=f"{args.save_metrics_folder}/table_order={experiment_order}_metric={metric}_seed={seed}.txt")


        print()
    
    save_obj = defaultdict(lambda: defaultdict(dict))

    for experiment_order, metric_seed_results in all_metric_scores.items():
        
        for metric, seed_results in metric_seed_results.items():

            # print(f"+++++++++++++++++ {metric} ++++++++++++++++++ ")
            # seed_results is a dict, key=seed, value=per_seed_metric_scores
            merged_results_across_seeds = merge_seed_results(seed_results)

            mean_results, std_results = calculate_mean_std(merged_results_across_seeds)

            save_obj[f"exp_order={experiment_order}"][metric]['mean_results'] = mean_results
            save_obj[f"exp_order={experiment_order}"][metric]['std_results'] = std_results

            print(f"******************* Avg scores for order={experiment_order}, metric={metric} *******************")
            pprint(mean_results)

            print(f"\n******************* Std scores for order={experiment_order}, metric={metric} *******************")
            pprint(std_results)
        save_obj[f"exp_order={experiment_order}"]['task_orders'] = all_task_orders[experiment_order]
        # save_obj[f"exp_order={experiment_order}"]['task_orders'] = all_task_orders

    # print(all_results_matrix)


    with open(f"{args.save_metrics_folder}/scores.json", 'w') as fp:
        json.dump(save_obj, fp)



def collect_multi_results(args, cl_stream):
    """ collect all results for MULTI_TASK learning """
    # seed_experiments = glob.glob(f"{path}/*")
    # print(seed_experiments)

    # for each seed in the experiments
    seed_experiments = glob.glob(f"{args.results_folder_dir}/*")

    # print(seed_experiments)

    all_metric_scores = defaultdict(lambda: defaultdict(dict))
    for seed_folder in seed_experiments:
        # get seed
        seed = seed_folder.split('seed')[1]
        seed = int(seed)
        print(f"========================== Collecting results for {cl_stream}, seed: {seed} ==========================")

        # if seed in [770, 688, 589, 983, 907]:
        # if seed in [589, 983, 907]:
        for metric in ['rougeL', 'rouge1', 'exact_match']:
        # for metric in ['rougeL']:
            print(f"******** Seed: {seed} Metric: {metric} ********")

            per_seed_metric_scores = prepare_one_seed_result_for_multi(f"{seed_folder}/results/", metric=metric)
            
            print(f"per_seed_metric_scores: {per_seed_metric_scores}")

            all_metric_scores[metric][f"seed={seed}"] = per_seed_metric_scores


        print()

    # print(all_metric_scores)

    save_obj = defaultdict(dict)
    for metric, seed_results in all_metric_scores.items():
        # print(f"+++++++++++++++++ {metric} ++++++++++++++++++ ")
        # seed_results is a dict, key=seed, value=per_seed_metric_scores

        # merge seed results by taking average
        merged_results_across_seeds = defaultdict(list)

        for seed, per_seed_result in seed_results.items():        
            for key, value in per_seed_result.items():
                merged_results_across_seeds[key].append(value)

        mean_results, std_results = calculate_mean_std(merged_results_across_seeds)

        save_obj[metric]['mean_results'] = mean_results
        save_obj[metric]['std_results'] = std_results

        print(f"******************* Avg scores for {metric} *******************")
        pprint(mean_results)

        print(f"\n******************* Std scores for {metric} *******************")
        pprint(std_results)

    with open(f"{args.save_metrics_folder}/scores.json", 'w') as fp:
        json.dump(save_obj, fp)



def main(args):

    if not os.path.isdir(args.save_metrics_folder):
        # if the output directory is not present then create it
        os.makedirs(args.save_metrics_folder)

    
    # get learning mode
    learning_mode_str = args.results_folder_dir.split('/stream')[0]
    print(learning_mode_str)

    cl_stream = args.results_folder_dir.split('stream=')[1]
    print(cl_stream)
    
    cl_method = None
    learning_mode = None
    if 'continual_instruction_tuning' in learning_mode_str:
        learning_mode = 'continual_instruction_tuning'
        
        # get cl method
        cl_method = args.results_folder_dir.split('CL=')[1]

        print(f"cl_method: {cl_method}")

        if cl_method == 'MULTI_TASK':
            print(f'yes')

            collect_multi_results(args, cl_stream)
        else:
            print(f'do CL parser')

            collect_cl_results(args, cl_stream, cl_method)

    elif 'initial_multitask_model' in learning_mode_str:
        learning_mode = 'initial_multitask_model'

    # cl_stream = args.results_folder_dir.split('stream=')[1]

    # print(cl_stream)

    # sys.exit(0)
    
    # # for each seed in the experiments
    # seed_experiments = glob.glob(f"{args.results_folder_dir}/*")

    # # all_metric_scores = defaultdict(dict)
    # all_metric_scores = defaultdict(lambda: defaultdict(dict))
    # all_task_orders = {}
    # all_results_matrix = defaultdict(dict)
    # for seed_folder in seed_experiments:
    #     # get seed
    #     seed = seed_folder.split('seed')[1]
    #     seed = int(seed)
    #     print(f"========================== Collecting results for {cl_stream}, seed: {seed} ==========================")

    #     # if learning_mode = 'continual_instruction_tuning':
    #     #     if cl_method == 'MULTI_TASK':
    #     #         collect_multi_results()
    #     #     else:
    #     #         collect_cl_results()


    #     sys.exit(0)


    #     for metric in ['rougeL', 'rouge1', 'exact_match']:
    #         results_matrix, ordered_cl_tasks, total_task_num, per_seed_metric_scores = prepare_one_seed_result_for_CL(
    #             f"{seed_folder}/results/", metric=metric)

    #         # print(f"per_seed_metric_scores: {per_seed_metric_scores}")

    #         # print_table(results_matrix, total_task_num, tablefmt='simple', write2file=False, save_path=None)

    #         print_table(results_matrix, total_task_num, tablefmt='simple', write2file=False, save_path=None)
    #         print_table(results_matrix, total_task_num, tablefmt='simple', write2file=True, 
    #             save_path=f"{args.save_metrics_folder}/table_metric={metric}_seed_{seed}.txt")
    #         print()

    #         all_metric_scores[metric][f"seed={seed}"] = per_seed_metric_scores
    #         all_results_matrix[metric][f"seed={seed}"] = results_matrix

    #     all_results_matrix[f"seed={seed}"] = results_matrix

    #     task_orders = [t[1] for t in ordered_cl_tasks]
    #     all_task_orders[f"seed={seed}"] = task_orders

    #     print()


    # save_obj = defaultdict(dict)
    # save_obj['task_orders'] = all_task_orders
    # for metric, seed_results in all_metric_scores.items():
    #     # print(f"+++++++++++++++++ {metric} ++++++++++++++++++ ")
    #     # seed_results is a dict, key=seed, value=per_seed_metric_scores
    #     merged_results_across_seeds = merge_seed_results(seed_results)

    #     mean_results, std_results = calculate_mean_std(merged_results_across_seeds)

    #     save_obj[metric]['mean_results'] = mean_results
    #     save_obj[metric]['std_results'] = std_results

    #     print(f"******************* Avg scores for {metric} *******************")
    #     pprint(mean_results)

    #     print(f"\n******************* Std scores for {metric} *******************")
    #     pprint(std_results)
    

    # with open(f"{args.save_metrics_folder}/scores.json", 'w') as fp:
    #     json.dump(save_obj, fp)





if __name__ == '__main__':

    args = parse_args()

    # assert args.initial_multitask_folder_dir != "", "you need to provide the initial multi-task trained model folder path"
    # assert args.results_folder_dir != "", "you need to provide the results folder path"

    main(args)