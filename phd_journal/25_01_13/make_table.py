# 1. Read results csv
# 2. Find F1-Score max and argmax PC for each Datasets
# 3. Make a double-entry table (Method, Datasets) with found F1-Score mean +- std

import pandas as pd
import numpy as np

# Read results csv
results = pd.read_csv('./25_01_13_results.csv')

# Find F1-Score max and argmax PC for each Datasets
datasets = results['Datasets'].unique()
methods = results['Method'].unique()
table = pd.DataFrame(index=methods, columns=datasets)
for dataset in datasets:
    dataset_results = results[results['Datasets'] == dataset]
    for method in methods:
        method_results = dataset_results[dataset_results['Method'] == method]
        f1_scores = method_results['Avg F1']
        max_f1_score = f1_scores.max()
        max_f1_score_pc = method_results[method_results['Avg F1'] == max_f1_score]['PCs'].values[0]
        table[dataset][method] = '{:.2f} ({:.2f})'.format(max_f1_score, max_f1_score_pc)

# Save table
table.to_csv('23_01_13_table.csv')
