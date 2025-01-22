# 1. Read results csv
# 2. Find F1-Score max and argmax PC for each Datasets
# 3. Make a double-entry table (Method, Datasets) with found F1-Score mean +- std

import pandas as pd
import numpy as np

# Read results csv
results = pd.read_csv('./25_01_16_results.csv')

# Find F1-Score max and argmax PC for each Datasets
datasets = results['Datasets'].unique()
methods = results['Method'].unique()

for n_pcs in range(2, 16):
    pcs_results = results[results['PCs'] == n_pcs]
    table = pd.DataFrame(index=methods, columns=datasets)

    for method in methods:
        method_results = pcs_results[pcs_results['Method'] == method]

        for dataset in datasets:
            dataset_results = method_results[method_results['Datasets'] == dataset]
            f1_score = dataset_results['Avg F1'].iloc[0]
            table[dataset][method] = '{:.2f}'.format(f1_score)

    # Save table
    table.to_csv(f'23_01_16_table_{n_pcs}pcs.csv')
