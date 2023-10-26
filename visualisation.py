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

    ax.set_ylim(depth, 0)
    ax.set_xlim(-depth/2, depth/2)

    # Move left y-axis and bottom x-axis to centre, passing through (0,0)
    ax.spines['left'].set_position('zero')
    ax.spines['top'].set_position('zero')

    # Eliminate upper and right axes
    ax.spines['right'].set_color('none')
    ax.spines['left'].set_color('none')
    ax.spines['top'].set_color('none')
    ax.spines['bottom'].set_color('none')



    plt.show()
    return

    if is_leaf_node(tree):
        # Plot a Leaf Label
        ax.text(0, 0, "Label: " + tree['value'],
                bbox = dict(facecolor='wheat', alpha=1, boxstyle='round'),
                ha='center', va='center')
        return
    else:
        # Plot a Node Label
        ax.text(0, 0, tree['attribute'] + " > " + tree['value'],
                bbox = dict(facecolor='wheat', alpha=1, boxstyle='round'),
                ha='center', va='center')
        # Plot 2 lines L and R
        plot_subtree(tree['left'])
        plot_subtree(tree['right'])

    plt.savefig(output_path + fname + ".png")


def plot_subtree(tree, depth, ax):
    plt.plot(x, y)
    return

if __name__ == "__main__":
    tree = {}
    with open("tree.pkl", "rb") as file:
        tree = pickle.load(file)
    plot_decision_tree(tree, 2, "test")
