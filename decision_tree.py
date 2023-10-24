#!/usr/bin/env python3

import numpy as np

# TODO: Define class Leaf and Tree

def find_split(dataset):
    # Chooses the attribute and the value that results in the highest information gain
    # (Defined in spec)
    return

def split_dataset(split, dataset):
    # Splits the dataset into 2 based on `split`
    return

def decision_tree_learning(training_dataset, depth):
    # Check if all samples have the same label
    # TODO: Can len(training_dataset) == 0 ever?
    if len(training_dataset) == 0 or all(s.label == training_dataset[0].label for s in training_dataset):
        return (Leaf(training_dataset[0].value), depth)
    else:
        split = find_split(training_dataset)
        node = DTree(split)
        l_branch, r_branch = split_dataset(training_dataset, split)
        l_branch, l_depth = decision_tree_learning(l_dataset, depth+1)
        r_branch, r_depth = decision_tree_learning(r_dataset, depth+1)
        return (node, max(l_depth, r_depth))

if __name__ == "__main__":
    clean_data = np.loadtxt("wifi_db/clean_dataset.txt")
    print("clean")
    print(clean_data)

    noisy_data = np.loadtxt("wifi_db/noisy_dataset.txt")
    print("noisy")
    print(noisy_data)
