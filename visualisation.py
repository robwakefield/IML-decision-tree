#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import pickle

output_path = "output/"

# Child Count Tree Node
# { 'value', 'left', 'right' }

# Checks if Tree Node is a Leaf Node
def is_leaf_node(node):
    return node['left'] == node['right'] == None

def get_attr_color(attr):
    color_list = ['bisque', 'limegreen', 'mediumslateblue', 'mistyrose', 'paleturquoise', 'plum', 'peachpuff']
    return color_list[int(attr)]

def get_label_color(label):
    color_list = ['bisque', 'limegreen', 'mediumslateblue', 'mistyrose', 'paleturquoise', 'plum', 'peachpuff']
    return color_list[int(label)]

def subtree_depths(tree):
    if is_leaf_node(tree):
        return {
                'left' : None,
                'right' : None,
                'value' : 1
                }
    else:
        ltree = subtree_depths(tree['left'])
        rtree = subtree_depths(tree['right'])
        return {
                'left' : ltree,
                'right' : rtree,
                'value' : max(ltree['value'], rtree['value']) + 1
                }

def plot_decision_tree(tree, depth, fname):
    # First form a new tree of similar dimensions where each node contains the depth of that subtree
    depth_tree = subtree_depths(tree)

    # Setup matplotlib
    fig = plt.figure(figsize=(50, 10))
    ax = plt.subplot(1, 1, 1)

    min_width = 5
    width = min_width * pow(2, depth)

    ax.set_ylim(depth, 0)
    ax.set_xlim(-width/2, width/2)

    # Move left y-axis and bottom x-axis to centre, passing through (0,0)
    ax.spines['left'].set_position('zero')
    ax.spines['top'].set_position('zero')

    # Eliminate upper and right axes
    ax.spines['right'].set_color('none')
    ax.spines['left'].set_color('none')
    ax.spines['top'].set_color('none')
    ax.spines['bottom'].set_color('none')

    # Remove ticks and tick labels
    ax.set_yticklabels([])
    ax.set_xticklabels([])
    ax.set_xticks([])
    ax.set_yticks([])

    # Call recursive helper function
    plot_subtree(tree, depth_tree, width/40, width/2, 0, 0, ax)

    #plt.show()
    plt.savefig(output_path + fname + ".png")

def plot_subtree(tree, depth_tree, min_width, offset, x, depth, ax):
    font_size = 5
    if depth < 3 :
        font_size = 10
    if depth < 10 :
        font_size = 6

    if is_leaf_node(tree):
        # Plot a Leaf Label
        plt.text(x, depth, "Label: " + str(tree['value']),
                fontsize=font_size,
                bbox = dict(facecolor=get_label_color(tree['value']), alpha=1, boxstyle='round'),
                ha='center', va='center')
    else:
        # Plot a Node Label
        plt.text(x, depth, "X" + str(tree['attribute']) + " > " + str(tree['value']),
                fontsize=font_size,
                bbox = dict(facecolor=get_attr_color(tree['attribute']), alpha=1, boxstyle='round'),
                ha='center', va='center')
        # Plot 2 lines L and R based on depth of subtrees
        ratio = depth_tree['left']['value'] / (depth_tree['left']['value'] + depth_tree['right']['value'])
        l = x - ratio * offset
        r = x + (1 - ratio) * offset
        # Let chains of Leaf Nodes appear close together
        if (is_leaf_node(depth_tree['right'])):
            l = x - min(min_width, 0.5 * offset)
        if (is_leaf_node(depth_tree['left'])):
            r = x + min(min_width, 0.5 * offset)
        
        plt.plot(np.array([x, l]), np.array([depth, depth + 1]), 'b')
        plt.plot(np.array([x, r]), np.array([depth, depth + 1]), 'r')
        # Recursively plot subtrees
        plot_subtree(tree['left'], depth_tree['left'], min_width, ratio * offset, l,  depth + 1, ax)
        plot_subtree(tree['right'], depth_tree['right'], min_width, (1 - ratio) * offset, r, depth + 1, ax)

if __name__ == "__main__":
    tree = {}
    with open("tree.pkl", "rb") as file:
        tree = pickle.load(file)
    plot_decision_tree(tree, 14, "test")
