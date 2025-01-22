import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from numpy import ndarray

results = pd.read_csv('./25_01_16_results.csv')
none_results = results[results['Method'] == 'none']

datasets = results['Datasets'].unique()
methods: ndarray = results['Method'].unique()
methods = methods[methods != 'none']  # Remove 'none' from methods
# x axis: pcs
# y axis: f1-score increase (method f1 - none f1)
# grouping: datasets

for method in methods:
    print("method: ", method)
    method_results = results[results['Method'] == method]

    res = pd.DataFrame(columns=['PCs', 'F1-Score Increase', 'Datasets'])

    for dataset in datasets:
        # Get
        dataset_method_results = method_results[method_results['Datasets'] == dataset]
        dataset_none_results = none_results[none_results['Datasets'] == dataset]
        # Sort them by PCs
        dataset_method_results = dataset_method_results.sort_values(by='PCs').reset_index(drop=True)
        dataset_none_results = dataset_none_results.sort_values(by='PCs').reset_index(drop=True)
        # Calculate f1-score increase
        f1_scores_diffs = dataset_method_results['Avg F1'] - dataset_none_results['Avg F1']
        pcs = dataset_method_results['PCs']
        res = res.append(pd.DataFrame({'PCs': pcs, 'F1-Score Increase': f1_scores_diffs, 'Datasets': dataset}))

    res = res.reset_index(drop=True)
    plt.figure(figsize=(13, 3))
    sns.set(style='whitegrid')
    ax = sns.barplot(res, x='PCs', y='F1-Score Increase', hue='Datasets')
    plt.title(method.title())
    plt.xlabel('Number of Features / PCs')
    plt.ylabel('F1-Score Increase')
    plt.ylim((-0.03, 0.085))
    # remove legend
    ax.get_legend().remove()
    #plt.show()
    # remove the plot frame lines
    ax.spines["top"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # secondary horizontal lines
    ax.yaxis.grid(True)
    ax.set_axisbelow(True)


    plt.savefig(f'25_01_16_plot_{method}.tiff', bbox_inches='tight', dpi=300)  # Save plot
    #
    plt.close()