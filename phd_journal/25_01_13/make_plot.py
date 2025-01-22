import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

results = pd.read_csv('./25_01_16_results.csv')

datasets = results['Datasets'].unique()
methods = results['Method'].unique()

for n_pcs in range(2, 16):
    pcs_results = results[results['PCs'] == n_pcs]

    # Bar plot
    sns.set(style='whitegrid')
    ax = sns.barplot(x='Datasets', y='Avg F1', hue='Method', data=pcs_results)
    ax.set_title(f'F1-Score for {n_pcs} PCs')
    ax.set_xlabel('Datasets')
    ax.set_ylabel('F1-Score')
    ax.legend(loc='lower right')
    # Save plot
    plt.savefig(f'25_01_16_plot_{n_pcs}pcs.png')
    plt.close()

