
from collections import defaultdict
import random
from tabulate import tabulate
import sys


def get_data_statistics(train_instances, dev_instances, test_instances):

    task2split2len = defaultdict(dict)

    for task_name, instances in train_instances.items():
        task2split2len[task_name]['train'] = len(instances)
    for task_name, instances in dev_instances.items():
        task2split2len[task_name]['dev'] = len(instances)
    for task_name, instances in test_instances.items():
        task2split2len[task_name]['test'] = len(instances)

    table = []
    for task_name, split_len in task2split2len.items():
        table.append({"task_name": task_name,
                      "train instances": split_len["train"],
                      "dev instances": split_len["dev"],
                      "test instances": split_len["test"]})
    print(tabulate(table, headers="keys"))
    
    # text_file = open("data_stat.csv", "w")
    # text_file.write(tabulate(table, headers="keys", tablefmt="tsv").replace('\t', ','))
    # text_file.close()


def get_task2instance(dataset):
    """ Return a new dict, key: task name, value: list of instances belong to the task
    dataset: list of dicts, each dict is an instance
    """

    # task to list of instances mapping
    task2instance = defaultdict(list)
    for instance in dataset:
        instance_task = instance["Task"]
        task2instance[instance_task].append(instance)

    task2instances_len = {task_name: len(instances) for task_name, instances in task2instance.items()}
    print(f"task2instances_len: {task2instances_len}")

    return task2instance


def train_dev_test_split_by_task(raw_datasets, max_num_instances_per_task, max_num_instances_per_eval_task, continual=False):
    """
    For each task, do train/dev/test split.
    We fix the number of dev instances equals the number of test instances, 
    which is set by max_num_instances_per_eval_task
    """

    # get all task names 
    all_task_names = set(i["Task"] for i in raw_datasets['train'])
    print(f"all_task_names: {len(all_task_names)}")

    # all_task_names = list(all_task_names)
    # ### random has 3 task orders
    # print(f"all_task_names: {all_task_names}")
    # random.shuffle(all_task_names)
    # print(f"order1: {all_task_names}")
    # print()

    # with open('data/CIT_data/task_orders/stream=cl_dialogue_long_tasks/order1.txt', 'w') as fout:
    #     for task in all_task_names:
    #         fout.write(task + "\n")

    # random.shuffle(all_task_names)
    # print(f"order2: {all_task_names}")
    # print()

    # with open('data/CIT_data/task_orders/stream=cl_dialogue_long_tasks/order2.txt', 'w') as fout:
    #     for task in all_task_names:
    #         fout.write(task + "\n")

    # random.shuffle(all_task_names)
    # print(f"order3: {all_task_names}")
    # print()

    # with open('data/CIT_data/task_orders/stream=cl_dialogue_long_tasks/order3.txt', 'w') as fout:
    #     for task in all_task_names:
    #         fout.write(task + "\n")

    # sys.exit(0)

    """ Remeber to record train/dev/test statistics for each task in CSV """
    task2instance = get_task2instance(raw_datasets['train'])

    # # task to list of instances mapping
    # task2instance = defaultdict(list)
    # for instance in raw_datasets['train']:
    #     instance_task = instance["Task"]
    #     task2instance[instance_task].append(instance)


    # when doing CL, we need to keep train/dev/test splits for each task individually
    if continual:
        # the key in below dicts are task_name
        train_instances, dev_instances, test_instances = defaultdict(list), defaultdict(list), defaultdict(list)

        # split each tasks' instances into train and test
        for task_name, instances in task2instance.items():
            test_instances[task_name].extend(instances[:max_num_instances_per_eval_task])
            dev_instances[task_name].extend(instances[max_num_instances_per_eval_task:max_num_instances_per_eval_task*2])

            # make sure per task training instances not exceeding the limit
            remaining_instances = instances[max_num_instances_per_eval_task*2:]
            print(f"total: {len(instances)}, remaining_instances: {len(remaining_instances)}")
            if len(remaining_instances) >= max_num_instances_per_task:
                random.shuffle(remaining_instances)
                train_instances[task_name].extend(remaining_instances[:max_num_instances_per_task])
            else:
                train_instances[task_name].extend(remaining_instances)

        print(f"dev_instances tasks: {len(dev_instances.keys())}")
        print(f"test_instances tasks: {len(test_instances.keys())}")
        print(f"train_instances tasks: {len(train_instances.keys())}")

        get_data_statistics(train_instances, dev_instances, test_instances)

        return train_instances, dev_instances, test_instances
        
    else:
        # when doing multi-task learning, after train/dev/test splits for each task, 
        # we can put them together as the mixed sets

        train_instances, dev_instances, test_instances = [], [], []
        # split each tasks' instances into train and test
        for task_name, instances in task2instance.items():
    
            test_instances.extend(instances[:max_num_instances_per_eval_task])
            dev_instances.extend(instances[max_num_instances_per_eval_task:max_num_instances_per_eval_task*2])

            # make sure per task training instances not exceeding the limit
            remaining_instances = instances[max_num_instances_per_eval_task*2:]
            print(f"total: {len(instances)}, remaining_instances: {len(remaining_instances)}")
            if len(remaining_instances) >= max_num_instances_per_task:
                random.shuffle(remaining_instances)
                train_instances.extend(remaining_instances[:max_num_instances_per_task])
            else:
                train_instances.extend(remaining_instances)
        
        print(f"dev_instances: {len(dev_instances)}")
        print(f"test_instances: {len(test_instances)}")
        print(f"train_instances: {len(train_instances)}")

        return train_instances, dev_instances, test_instances


def get_replay_instances_by_task(multitask_train_dataset, replay_num_instance_per_task):

    task2instance = get_task2instance(multitask_train_dataset)

    replay_instances = []
    for task_name, instances in task2instance.items():
        # print(f"total: {len(instances)}, replay_num_instance_per_task: {replay_num_instance_per_task}")
        if len(instances) >= replay_num_instance_per_task:
            replay_instances.extend(instances[:replay_num_instance_per_task])
        else:
            replay_instances.extend(instances)
    print(f"replay_instances: {len(replay_instances)}")

    return replay_instances