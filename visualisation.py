#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import pickle

output_path = "output/"

# Checks if Tree Node is a Leaf Node
def is_leaf_node(node):
    return node['left'] == node['right'] == None

def plot_decision_tree(tree, depth, fname):
    fig, ax = plt.subplots()

    width = 100

    ax.set_ylim(depth, 0)
    ax.set_xlim(-width, width)

    # Move left y-axis and bottom x-axis to centre, passing through (0,0)
    ax.spines['left'].set_position('zero')
    ax.spines['top'].set_position('zero')

    # Eliminate upper and right axes
    ax.spines['right'].set_color('none')
    ax.spines['left'].set_color('none')
    ax.spines['top'].set_color('none')
    ax.spines['bottom'].set_color('none')

    plot_subtree(tree, 0, 0, ax)

    plt.show()
    plt.savefig(output_path + fname + ".png")

def get_attr_color(attr):
    color_list = ['bisque', 'limegreen', 'mediumslateblue', 'mistyrose', 'paleturquoise', 'plum', 'peachpuff']
    return color_list[int(attr)]

def get_label_color(label):
    color_list = ['bisque', 'limegreen', 'mediumslateblue', 'mistyrose', 'paleturquoise', 'plum', 'peachpuff']
    return color_list[int(label)]

def plot_subtree(tree, x, depth, ax):
    if is_leaf_node(tree):
        # Plot a Leaf Label
        ax.text(x, depth, "Label: " + str(tree['value']),
                fontsize=6,
                bbox = dict(facecolor=get_label_color(tree['value']), alpha=1, boxstyle='round'),
                ha='center', va='center')
    else:
        # Plot a Node Label
        ax.text(x, depth, "X" + str(tree['attribute']) + " > " + str(tree['value']),
                fontsize=6,
                bbox = dict(facecolor=get_attr_color(tree['attribute']), alpha=1, boxstyle='round'),
                ha='center', va='center')
        # Plot 2 lines L and R
        l = x-(30/(depth+1)) - 4
        r = x+(30/(depth+1)) + 4
        plt.plot(np.array([x, l]), np.array([depth, depth+1]), 'b')
        plt.plot(np.array([x, r]), np.array([depth, depth+1]), 'r')
        plot_subtree(tree['left'], l,  depth + 1, ax)
        plot_subtree(tree['right'], r, depth + 1, ax)


    return

if __name__ == "__main__":
    tree = {}
    with open("tree.pkl", "rb") as file:
        tree = pickle.load(file)
    plot_decision_tree(tree, 14, "test")
