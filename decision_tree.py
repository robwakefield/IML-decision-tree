#!/usr/bin/env python3

import numpy as np

# Dataset: 2000x8 array
#  [ sig_1, sig_2, sig_3, sig_4, sig_5, sig_6, sig_7, label]x2000

# Tree Node: (A Leaf Node just has a value. 'attribute', 'left' and 'right' are None)
#   { 'attribute', 'value', 'left', 'right' }

def get_label(sample):
    # Readable function to return the label of a sample
    return sample[7]

def find_split(dataset):
    # Chooses the attribute and the value that results in the highest information gain
    # (Defined in spec)
    # TODO: implement
    return (attr, value)

def split_dataset(dataset, split_attr, split_val):
    # Splits the dataset into 2 based on 'split_attr' and 'split_val'
    # TODO: implement
    return (l_dataset, r_dataset)

def decision_tree_learning(training_dataset, depth):
    # Check if all samples have the same label
    # TODO: Can len(training_dataset) == 0 ever?
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
                'left': l_branch,
                'right': r_branch
                }
    
        return (node, max(l_depth, r_depth))

if __name__ == "__main__":
    clean_data = np.loadtxt("wifi_db/clean_dataset.txt")
    print("clean")
    print(clean_data)

    noisy_data = np.loadtxt("wifi_db/noisy_dataset.txt")
    print("noisy")
    print(noisy_data)
