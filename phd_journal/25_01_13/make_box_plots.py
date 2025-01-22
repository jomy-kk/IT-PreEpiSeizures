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

fig = plt.figure(figsize=(10, 4))
sns.set(style='whitegrid')
ax = sns.boxplot(results, x='Datasets', y='Avg F1', hue='Method')
plt.xlabel('Number of Datasets')
plt.ylabel('Weighted F1-Score')
# remove legend
ax.get_legend().remove()
#plt.show()
# remove the plot frame lines
ax.spines["top"].set_visible(False)
#ax.spines["bottom"].set_visible(False)
ax.spines["right"].set_visible(False)

# secondary horizontal lines
ax.yaxis.grid(True)
ax.set_axisbelow(True)


plt.savefig('25_01_16_boxplot.pdf', bbox_inches='tight', dpi=300)  # Save plot
#
plt.close()