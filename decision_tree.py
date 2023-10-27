#!/usr/bin/env python3

import numpy as np
import math
import random
import sys

from visualisation import plot_decision_tree

# Dataset: 2000x8 array
#  [ sig_1, sig_2, sig_3, sig_4, sig_5, sig_6, sig_7, label ]x2000

# Tree Node: (A Leaf Node just has a value. 'attribute', 'left' and 'right' are None)
#   { 'attribute', 'value', 'left', 'right' }

labels = []

def label_to_index(label):
    return np.where(labels == label)[0]

def no_of_labels():
    return len(labels)

def register_labels(dataset):
    global labels
    labels = np.unique(dataset[:, -1])
    # print(no_of_labels())

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
    
def create_decision_tree(dataset):
    return decision_tree_learning(dataset, 0)
    
def evaluate_data(test_data, tree):
    if is_leaf_node(tree):
        return (get_label(test_data), tree['value'])
    if test_data[tree['attribute']] > tree['value']:
        return evaluate_data(test_data, tree['right'])
    else:
        return evaluate_data(test_data, tree['left'])

def evaluate(test_dataset, tree):
    confusion_mat = np.zeros((no_of_labels(), no_of_labels()))
    # sorted_dataset = test_dataset[np.argsort(test_dataset[:, -1])]
    for test in test_dataset:
        actual, predict = evaluate_data(test, tree)
        confusion_mat[label_to_index(actual), label_to_index(predict)] += 1
    print(confusion_mat)
    return confusion_mat
    
def cal_accuracy(confusion_matrix):
    T = np.sum(np.diag(confusion_matrix))
    total = np.sum(confusion_matrix)
    assert(total > 0)
    return T / total
    
# validation set taken from training set
def prune_tree(validation_set, tree):
    return test_tree_for_pruning(validation_set, tree, 0)
    
def test_tree_for_pruning(validation_set, tree, depth):
    if (validation_set.shape[0] == 0):
    	return (tree, np.zeros((no_of_labels(), no_of_labels())), depth)
        
    if (is_leaf_node(tree)):
    	# Get confusion matrix on the leaf node
    	confusion_matrix = evaluate(validation_set, tree)
    	return (tree, confusion_matrix, depth)
        
    left_validation_set, right_validation_set = split_dataset(validation_set, tree['attribute'], tree['value'])
    new_left_tree, left_confusion_matrix, left_depth = test_tree_for_pruning(left_validation_set, tree['left'], depth + 1)
    new_right_tree, right_confusion_matrix, right_depth = test_tree_for_pruning(right_validation_set, tree['right'], depth + 1)    

    new_tree = {
    	'attribute' : tree['attribute'],
    	'value': tree['value'],
    	'left' : new_left_tree,
    	'right' : new_right_tree
    }

    if ((not is_leaf_node(tree['left'])) or (not is_leaf_node(tree['right']))):
    	return (new_tree, None, max(left_depth, right_depth))
    	
    # Compare accuracy for prune and non-prune tree

    # Find original accuracy
    merged_confusion_matrix = left_confusion_matrix + right_confusion_matrix
    original_accuracy = cal_accuracy(merged_confusion_matrix)
 
    # Find pruned accuracy 
    majority_label = np.bincount(validation_set[:, -1].astype(int)).argmax()
    pruned_current_tree = {
			'attribute' : None,
			'value': majority_label,
			'left' : None,
			'right' : None
			}
    pruned_confusion_matrix = evaluate(validation_set, pruned_current_tree)
    pruned_accuracy = cal_accuracy(pruned_confusion_matrix)

    if (original_accuracy >= pruned_accuracy):
        # Don't prune
	    assert (left_depth == right_depth)
	    return (new_tree, merged_confusion_matrix, max(left_depth, right_depth))
    else:
        # Prune
        print('PRUNED')
        new_depth = depth - 1
        print(pruned_current_tree, pruned_confusion_matrix, new_depth)
        return (pruned_current_tree, pruned_confusion_matrix, new_depth)
    	

# split dataset into folds, array of (testing_set, training_set)
def split_folds_dataset(dataset, fold_no):
    np.random.default_rng().shuffle(dataset)
    folds = [fold.tolist() for fold in np.array_split(dataset, fold_no)]
    fold_list = []
    for i, f in enumerate(folds):
        training_list = np.array([j for s in folds[:i] + folds[i+1:] for j in s])
        fold_list.append((np.array(f), training_list))
        print(f"test set", np.array(f), "\ntraining set\n", training_list)
    return fold_list

def cross_validation(dataset, fold_no):
    fold_list = split_folds_dataset(dataset, fold_no)
    tree_list = []
    depth_list = []
    confusion_matrix_list = []
    accuracy_list =[]
    for i, (test_dataset, training_dataset) in enumerate(fold_list):
        (tree, depth) = create_decision_tree(training_dataset)
        # print(tree, depth)
        confusion_matrix = evaluate(test_dataset, tree)
        accuracy = cal_accuracy(confusion_matrix)
        tree_list.append(tree)
        depth_list.append(depth)
        confusion_matrix_list.append(confusion_matrix)
        accuracy_list.append(accuracy)
        print(f"Confusion Matrix of Fold {i}:\n", confusion_matrix, f"Accuracy:", accuracy)
    best_fold = accuracy_list.index(max(accuracy_list))
    print(confusion_matrix_list[best_fold], accuracy_list[best_fold])
    return (tree_list, depth_list, confusion_matrix_list, accuracy_list, best_fold)

if __name__ == "__main__":
    clean_dataset = np.loadtxt("wifi_db/clean_dataset.txt")
    # print("clean dataset")
    # print(clean_dataset)

    noisy_dataset = np.loadtxt("wifi_db/noisy_dataset.txt")
    # print("noisy dataset")
    # print(noisy_dataset)

    small_dataset = np.loadtxt("wifi_db/small_dataset.txt")
    # print("Data", small_dataset)
    # find_split(small_dataset)

    # Run input dataset if exists
    if len(sys.argv) > 1:
        input_dataset = np.loadtxt(sys.argv[1])
        print("Input dataset: ", sys.argv[1])
        register_labels(input_dataset)
        trees, depths, confusion_matrices, accuracies, fold_no = cross_validation(input_dataset, 10)

        # TODO: anything else to add?

    else:
        print("Clean dataset: ")
        register_labels(clean_dataset)
        trees, depths, confusion_matrices, accuracies, fold_no = cross_validation(clean_dataset, 10)
        
        # tree, depth = create_decision_tree(clean_dataset)
        # plot_decision_tree(tree, depth, "clean_tree")
        # print("depth:", depth)

        # validation set taken from training set
        # pruned_tree, _, pruned_depth = prune_tree(clean_dataset, tree)
        # plot_decision_tree(pruned_tree, pruned_depth, "pruned_tree")
        # print("depth:", pruned_depth)

        # # Testing
        # print(cal_entropy(small_dataset))

