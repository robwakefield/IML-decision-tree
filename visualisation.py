#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import pickle

output_path = "output/"

# Checks if Tree Node is a Leaf Node
def is_leaf_node(node):
    return node['left'] == node['right'] == None

def get_attr_color(attr):
    color_list = ['bisque', 'limegreen', 'mediumslateblue', 'mistyrose', 'paleturquoise', 'plum', 'peachpuff']
    return color_list[int(attr)]

def get_label_color(label):
    color_list = ['bisque', 'limegreen', 'mediumslateblue', 'mistyrose', 'paleturquoise', 'plum', 'peachpuff']
    return color_list[int(label)]

def plot_decision_tree(tree, depth, fname):
    fig, ax = plt.subplots()

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

    plot_subtree(tree, min_width, depth, 0, 0, ax)

    plt.show()
    #plt.savefig(output_path + fname + ".png")

def plot_subtree(tree, min_width, max_depth, x, current_depth, ax):
    font_size = 6
    if is_leaf_node(tree):
        # Plot a Leaf Label
        ax.text(x, current_depth, "Label: " + str(tree['value']),
                fontsize=font_size,
                bbox = dict(facecolor=get_label_color(tree['value']), alpha=1, boxstyle='round'),
                ha='center', va='center')
    else:
        # Plot a Node Label
        ax.text(x, current_depth, "X" + str(tree['attribute']) + " > " + str(tree['value']),
                fontsize=font_size,
                bbox = dict(facecolor=get_attr_color(tree['attribute']), alpha=1, boxstyle='round'),
                ha='center', va='center')
        # Plot 2 lines L and R
        n = current_depth - max_depth
        offset = pow(2, -n - 2) * min_width
        l = x - offset
        r = x + offset
        plt.plot(np.array([x, l]), np.array([current_depth, current_depth + 1]), 'b')
        plt.plot(np.array([x, r]), np.array([current_depth, current_depth + 1]), 'r')
        # Recursively plot subtrees
        plot_subtree(tree['left'], min_width, max_depth, l,  current_depth + 1, ax)
        plot_subtree(tree['right'], min_width, max_depth, r, current_depth + 1, ax)

if __name__ == "__main__":
    tree = {}
    with open("tree.pkl", "rb") as file:
        tree = pickle.load(file)
    plot_decision_tree(tree, 14, "test")
