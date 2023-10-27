# Introduction to Machine Learning - Decision Tree Coursework

Based on WIFI signal strengths collected from a mobile phone, a decision tree algorithm was implemented to determine its indoor location.

## Usage

- Option 1: Default datasets ("_WIFI_db/clean dataset.txt_" and "_WIFI_db/noisy dataset.txt_")

    Run **python decision_tree.py**

    This will create a decision tree for the clean and noisy datasets.

    The program will output the decision trees as output/clean_tree.png and output/noisy_tree.png.

- Option 2: New dataset

    Run **python decision_tree.py \<**_path to dataset_**>**

    This will create a decision tree for the given dataset.

    The program will output 2 versions of the decision tree:
    1. output/dataset_tree.png: A symmetrical binary tree image.
    2. output/dataset_tree_alternative.png: An asymmetrical tree that may display better for high depth trees.

