#!/usr/bin/env python3

import numpy as np

clean_data = np.loadtxt("wifi_db/clean_dataset.txt")
print("clean")
print(clean_data)

noisy_data = np.loadtxt("wifi_db/noisy_dataset.txt")
print("noisy")
print(noisy_data)
