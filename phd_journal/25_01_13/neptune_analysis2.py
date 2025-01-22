import pandas as pd
import numpy as np

# get pwd
import os
print(os.getcwd())

x = pd.read_csv("./25_01_16.csv")
x = x.iloc[:-49]
res = []

for n_datasets in (2, 3, 4, 5, 6):
    y = x[x['Datasets'].apply(lambda x: len(eval(x)) == n_datasets)]
    print("Number of datasets:", n_datasets)

    methods = ('none', 'neuroharmonize', 'neuroharmonize2', 'nestedcombat+', 'nestedcombat-')
    for method in methods:
        print("Method:", method)
        y_method = y[y['Method'] == method]
        pass

        for pcs in y_method['Features'].unique():
            n_pc = len(eval(pcs))
            print("Number of PCs:", n_pc)
            y_method_pcs = y_method[y_method['Features'] == pcs]

            avg_f1 = y_method_pcs['F1-Score'].mean()
            std_f1 = y_method_pcs['F1-Score'].std()

            res.append({
                'Method': method,
                'Datasets': n_datasets,
                'PCs': n_pc,
                'Avg F1': avg_f1,
                'Std F1': std_f1
            })
            pass
res = pd.DataFrame(res)
res.to_csv("25_01_16_results.csv", index=False)
