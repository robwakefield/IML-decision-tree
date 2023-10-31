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

# Initialise labels
def register_labels(dataset):
    global labels
    labels = np.unique(dataset[:, -1])

# Get index of label
def label_to_index(label):
    return np.where(labels == label)[0]

# Get number of labels
def no_of_labels():
    return len(labels)

# Readable function to return the label of a sample
def get_label(sample):
    return sample[-1]

# Checks if Tree Node is a Leaf Node
def is_leaf_node(node):
    return node['left'] == node['right'] == None

# Chooses the attribute and the value that results in the highest information gain
def find_split(dataset):
    # Store all information gain of different attributes and values 
    # info_gain_dict = {(attribute, value): info_gain}
    info_gain_dict = {}

    for attr in range(dataset.shape[1] - 1):
        # Sort on attribute value
        sorted_dataset = dataset[np.argsort(dataset[:, attr])]
        for row in range(sorted_dataset.shape[0] - 1):
            # split_value is the average of the values of two examples on a single attribute
            split_value = (sorted_dataset[row, attr] + sorted_dataset[row + 1, attr]) / 2

            # Split the dataset into two according to the split point
            left, right = split_dataset(dataset, attr, split_value)

            # Store the info gain of the current split point to info_gain_dict
            info_gain_dict[(attr, split_value)] = cal_info_gain(dataset, left, right)

    # The split point with the maximum info gain will be selected
    (selected_attr, selected_value) = max(info_gain_dict, key=lambda k: info_gain_dict[k])
    return (selected_attr, selected_value)

# Calculate entropy of dataset
def cal_entropy(dataset):
    labels = dataset[:, -1]

    # counts = [c1, c2, c3, ...] where ci is the number of examples of class i
    counts = np.unique(labels, return_counts=True)[1]

    # prob = [p1, p2, p3, ...] where pi is probability of class i in this dataset
    prob = counts / dataset.shape[0]

    # Entropy = -SUM(pi * log2(pi))
    entropy = 0
    for p in prob:
        entropy -= p * math.log2(p)
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

# Recursive helper function to create the decision tree
def decision_tree_learning(training_dataset, depth, prev_left_size=None, prev_right_size=None):
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

        # If find_split() fails to split, return leaf node to stop recursion
        # (E.g. when all attributes are the same)
        if (prev_left_size == len(l_dataset) and prev_right_size == len(r_dataset)):
            # Construct Leaf Node
            return ({
                    'attribute' : None,
                    'value': get_label(training_dataset[0]),
                    'left' : None,
                    'right' : None
                    }, depth)

        # Recursively run decision tree on L and R branches
        l_branch, l_depth = decision_tree_learning(l_dataset, depth+1, len(l_dataset), len(r_dataset))
        r_branch, r_depth = decision_tree_learning(r_dataset, depth+1, len(l_dataset), len(r_dataset))
        
        # Construct the tree with root node as split value
        node = {
                'attribute' : split_attr, # e.g. X0, X4
                'value' : split_val, # e.g. -55.5, -70.5
                'left' : l_branch,
                'right' : r_branch
                }
    
        return (node, max(l_depth, r_depth))
    
# Inital function for creating decision tree
def create_decision_tree(dataset):
    return decision_tree_learning(dataset, 0)
    
# Give result to one example in a tuple, (Actual Class, Predicted Class)
def evaluate_data(test_data, tree):
    if is_leaf_node(tree):
        return (get_label(test_data), tree['value'])
    if test_data[tree['attribute']] > tree['value']:
        return evaluate_data(test_data, tree['right'])
    else:
        return evaluate_data(test_data, tree['left'])

# Give the confusion matrix of the dataset on a tree
def evaluate(test_dataset, tree):
    confusion_mat = np.zeros((no_of_labels(), no_of_labels()))
    for test in test_dataset:
        actual, predict = evaluate_data(test, tree)
        confusion_mat[label_to_index(actual), label_to_index(predict)] += 1
    return confusion_mat
    
# Prune tree from validation set (NOT MARKED)
def prune_tree(validation_set, tree):
    return test_tree_for_pruning(validation_set, tree, 0)
    
# Recursive helper function for prune_tree (NOT MARKED)
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
        new_depth = depth - 1
        return (pruned_current_tree, pruned_confusion_matrix, new_depth)
    	

# Split dataset into folds, array of (testing_set, training_set)
def split_folds_dataset(dataset, fold_no):
    seed = 22
    np.random.default_rng(22).shuffle(dataset)
    folds = [fold.tolist() for fold in np.array_split(dataset, fold_no)]
    fold_list = []
    for i, f in enumerate(folds):
        training_list = np.array([j for s in folds[:i] + folds[i+1:] for j in s])
        fold_list.append((np.array(f), training_list))
    return fold_list

# does FOLD_NO-fold cross validation and returns all folds and the fold with highest accuracy
def cross_validation(dataset, fold_no):
    fold_list = split_folds_dataset(dataset, fold_no)
    tree_list = []
    depth_list = []
    confusion_matrix_list = []
    accuracy_list =[]
    for i, (test_dataset, training_dataset) in enumerate(fold_list):
        
        progress = "#" * i
        print(progress+"\r", end="")

        (tree, depth) = create_decision_tree(training_dataset)
        confusion_matrix = evaluate(test_dataset, tree)
        accuracy = cal_accuracy(confusion_matrix)
        tree_list.append(tree)
        depth_list.append(depth)
        confusion_matrix_list.append(confusion_matrix)
        accuracy_list.append(accuracy)
    best_fold = accuracy_list.index(max(accuracy_list))
    return (tree_list, depth_list, confusion_matrix_list, best_fold)

# Get the accuracy of each class from a confusion matrix
def cal_accuracy(confusion_matrix):
    correct_examples = np.sum(np.diag(confusion_matrix))
    total = np.sum(confusion_matrix)
    assert(total > 0)
    return correct_examples / total

# Get the precision of each class from a confusion matrix
def cal_precision(confusion_matrix, i):
    tp = confusion_matrix[i][i]
    fp = np.sum(confusion_matrix[:, i]) - tp
    return tp / (tp + fp)

# Get the recall of each class from a confusion matrix
def cal_recall(confusion_matrix, i):
    tp = confusion_matrix[i][i]
    fn = np.sum(confusion_matrix[i]) - tp
    return tp / (tp + fn)

# Get the F1_measure of each class from a confusion matrix
def cal_F1_measure(confusion_matrix, i):
    recall = cal_recall(confusion_matrix, i)
    precision = cal_precision(confusion_matrix, i)
    return (2 * precision * recall) / (precision + recall)

def print_metrics(confusion_matrix):
    print("Confusion Matrix:\n", confusion_matrix)
    print("Accuracy:", cal_accuracy(confusion_matrix))

    total_precision = 0
    total_recall = 0
    total_f1 = 0

    for i in range(no_of_labels()):
        precision = cal_precision(confusion_matrix, i)
        recall = cal_recall(confusion_matrix, i)
        f1 = cal_F1_measure(confusion_matrix, i)

        total_precision += precision
        total_recall += recall
        total_f1 += f1

        print(f"Class {i+1}: Precision: {precision} Recall: {recall} F1-Measure: {f1}")

    print("Macro-Averaged Precision:", total_precision / no_of_labels())
    print("Macro-Averaged Recall:", total_recall / no_of_labels())
    print("Macro-Averaged F1:", total_f1 / no_of_labels())

folds = 10

if __name__ == "__main__":

    if len(sys.argv) != 2:
        print("Invalid number of arguments")
    else:
        input_dataset = np.loadtxt(sys.argv[1])
        register_labels(input_dataset)

        # Create a decision tree from the input_dataset
        print('Training on the Input Dataset...')
        tree, depth = create_decision_tree(input_dataset)

        # Create an image of the tree
        print('Ploting Decision Tree...')
        plot_decision_tree(tree, depth, "dataset_tree", depth_based=False)
        plot_decision_tree(tree, depth, "dataset_tree_alternative", depth_based=True)

        # Test the model using cross vaidation
        print("Running Cross Validation...")
        trees, depths, confusion_matrices, best_fold_no = cross_validation(input_dataset, folds)
        print('Result after cross validation:')
        print_metrics(sum(confusion_matrices))
