#!/usr/bin/env python3

import numpy as np
import math
import random

# Dataset: 2000x8 array
#  [ sig_1, sig_2, sig_3, sig_4, sig_5, sig_6, sig_7, label ]x2000

# Tree Node: (A Leaf Node just has a value. 'attribute', 'left' and 'right' are None)
#   { 'attribute', 'value', 'left', 'right' }

no_of_rooms = -1

# Readable function to return the label of a sample
def get_label(sample):
    return sample[-1]

# Checks if Tree Node is a Leaf Node
def is_leaf_node(node):
    return node['left'] == node['right'] == None

# Chooses the attribute and the value that results in the highest information gain
def find_split(dataset):
    # Store all information gain of different attributes and values
    info_gain = {}

    for attr in range(dataset.shape[1] - 1):
        # Sort on attribute value
        sorted_dataset = dataset[np.argsort(dataset[:, attr])]
        for row in range(sorted_dataset.shape[0] - 1):
            if sorted_dataset[row, -1] == sorted_dataset[row + 1, -1]:
                continue
            # Average of two values
            value = (sorted_dataset[row, attr] + sorted_dataset[row + 1, attr]) / 2
            left, right = split_dataset(dataset, attr, value)
            info_gain[(attr, value)] = cal_info_gain(dataset, left, right)

    (selected_attr, selected_value) = max(info_gain, key=lambda k: info_gain[k])
    return (selected_attr, selected_value)

# Calculate entropy of dataset
def cal_entropy(dataset):
    rooms = dataset[:, -1]
    counts = np.unique(rooms, return_counts=True)[1]
    room_prob = counts / dataset.shape[0]
    entropy = 0
    for p in room_prob:
        entropy += -p * math.log2(p)
    return entropy

# Calculate information gain after spitting original dataset into left and right
def cal_info_gain(original, left, right):
    remainder = left.size * cal_entropy(left) + right.size * cal_entropy(right)
    remainder /= original.size
    return cal_entropy(original) - remainder


# Splits the dataset into 2 based on 'split_attr' and 'split_val'
# Assume that r_dataset <- attr > value
#         and l_dataset <- attr <= value
def split_dataset(dataset, attr, value):
    l_dataset = dataset[dataset[:, attr] <= value]
    r_dataset = dataset[dataset[:, attr] > value]
    return (l_dataset, r_dataset)

def decision_tree_learning(training_dataset, depth):
    if len(training_dataset) == 0:
        return ({
                'attribute' : None,
                'value': -1,
                'left' : None,
                'right' : None
                }, depth)

    # Check if all samples have the same label
    if all(get_label(s) == get_label(training_dataset[0]) for s in training_dataset):
        # Construct Leaf Node
        return ({
                'attribute' : None,
                'value': get_label(training_dataset[0]),
                'left' : None,
                'right' : None
                }, depth)
    else:
        # Identify and split database to maximise information gain
        split_attr, split_val = find_split(training_dataset)
        l_dataset, r_dataset = split_dataset(training_dataset, split_attr, split_val)

        # Recursively run decision tree on L and R branches
        l_branch, l_depth = decision_tree_learning(l_dataset, depth+1)
        r_branch, r_depth = decision_tree_learning(r_dataset, depth+1)
        
        # Construct the tree with root node as split value
        node = {
                'attribute' : split_attr, # e.g. X0, X4
                'value' : split_val, # e.g. -55.5, -70.5
                'left' : l_branch,
                'right' : r_branch
                }
    
        return (node, max(l_depth, r_depth))
    
def evaluate_data(test_data, tree):
    if is_leaf_node(tree):
        return (get_label(test_data), tree['value'])
    if test_data[tree['attribute']] > tree['value']:
        return evaluate_data(test_data, tree['right'])
    else:
        return evaluate_data(test_data, tree['left'])

def evaluate(test_dataset, tree):
    confusion_mat = np.zeros((no_of_rooms, no_of_rooms))
    sorted_dataset = test_dataset[np.argsort(test_dataset[:, -1])]
    for test in test_dataset:
        # actual, predict = evaluate_data(test, tree)
        confusion_mat[evaluate_data(test, tree)] += 1
    print(confusion_mat)
    return confusion_mat


# TODO: split dataset into 10 folds
# def split_dataset(dataset, test_proportion, random_generator = random.default_rng()):
#     random_generator.shuffle(dataset, axis=0)
#     test_size = int(dataset.shape[0] * test_proportion)
#     train_size = dataset.shape[0] - test_size
#     return (data[:train_size, :-1], data[:test_size, :-1], data[:train_size, -1], data[:test_size, -1])


if __name__ == "__main__":
    clean_dataset = np.loadtxt("wifi_db/clean_dataset.txt")
    print("clean dataset")
    print(clean_dataset)

    noisy_dataset = np.loadtxt("wifi_db/noisy_dataset.txt")
    print("noisy dataset")
    print(noisy_dataset)

    # small_dataset = np.loadtxt("wifi_db/small_dataset.txt")
    # print("Data", small_dataset)
    # find_split(small_dataset)
    
    no_of_rooms = np.unique(clean_dataset[:, -1])
    tree, depth = decision_tree_learning(clean_dataset, 0)
    print(tree)
    print("depth:", depth)

    # # Testing
    # print(cal_entropy(small_dataset))

