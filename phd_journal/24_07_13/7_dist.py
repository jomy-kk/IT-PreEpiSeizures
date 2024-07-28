# We will compare the distribution of 6 features in two populations.
# Each population will have 3 groups.
# At the end we want violin plots with seaborn.

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from read import read_elders, feature_names, read_children
from utils import curate_feature_names

# Get elders and children data
features_elders, targets_elders = read_elders()
features_children, targets_children = read_children()

# Groups with targets boundaries
#age_classes = ((0,8), (15, 20))
age_classes = ((0,6), (6, 10), (10, 13), (13, 20))
#mmse_classes = ((0,15), (24, 31))
mmse_classes = ((0,12), (13, 16), (17,24), (25, 30))

# Assign groups and population
features_elders["Group"] = np.nan
features_elders["Population"] = np.nan
features_children["Group"] = np.nan
features_children["Population"] = np.nan

for i, (start, end) in enumerate(age_classes):
    features_children.loc[(targets_children >= start) & (targets_children <= end), "Group"] = i
    features_children.loc[(targets_children >= start) & (targets_children <= end), "Population"] = "Children"

for i, (start, end) in enumerate(mmse_classes):
    features_elders.loc[(targets_elders >= start) & (targets_elders <= end), "Group"] = i
    features_elders.loc[(targets_elders >= start) & (targets_elders <= end), "Population"] = "Elders"

# Drop NaN values
features_elders.dropna(inplace=True)
features_children.dropna(inplace=True)

# Features to compare

FEATURES = [
    "COH#Frontal(R)-Temporal(L)#theta",
]

#FEATURES = feature_names

for feature in FEATURES:
    print("=====================================")
    print("Feature: {}".format(feature))

    # Select only important features
    features_elders_ = features_elders.loc[:, (feature, "Group", "Population")]
    features_children_ = features_children.loc[:, (feature, "Group", "Population")]

    # Concatenate data
    features = pd.concat([features_elders_, features_children_])

    # Check with t-test if the distributions are different group pair-wise, each population in separate
    # We will use the t test
    from scipy.stats import ttest_ind, ks_2samp

    # Get the groups
    groups = features["Group"].unique()
    populations = features["Population"].unique()

    outliers = []
    p_values = []
    for population in populations:
        for group1 in groups:
            for group2 in groups:
                if group1 == group2:
                    continue
                else:
                    # Get the data
                    data1 = features.loc[(features["Group"] == group1) & (features["Population"] == population), feature]
                    data2 = features.loc[(features["Group"] == group2) & (features["Population"] == population), feature]
                    # Select 10% outliers by 2 standard deviations
                    outliers.extend(data1[(data1 < data1.mean() - 1.5*data1.std()) | (data1 > data1.mean() + 1.5*data1.std())].index)
                    outliers.extend(data2[(data2 < data2.mean() - 1.5*data2.std()) | (data2 > data2.mean() + 1.5*data2.std())].index)

    for group in groups:
        # get data of the group
        data = [features.loc[(features["Group"] == group) & (features["Population"] == population), feature] for population in populations]
        # Perform a Mann-Whitney U tes test to determine if the distribution is similar between the two populations
        statistic, pvalue = ks_2samp(*data)
        print("Group: {}, KS: {}, p-value: {}".format(group, statistic, pvalue))


    # Perform ANOVA test among the four groups, for each population separately.
    from scipy.stats import f_oneway

    for population in populations:
        # separate 4 groups
        data = [features.loc[(features["Group"] == group) & (features["Population"] == population), feature] for group in groups]
        statistic, pvalue = f_oneway(*data)
        print("Population: {}, ANOVA p-value: {}".format(population, pvalue))


    # Make violins plot
    # in the x-axis we will have 3 groups, each with two classes, elders and children, so 6 violins
    # in the y-axis we will have the feature values
    plt.rcParams['font.family'] = 'Arial'
    plt.rcParams['font.size'] = 12
    fig, ax = plt.subplots()
    # make grid only horizontal
    plt.grid(True, which='major', linestyle='--', linewidth=0.5, color="#CCCCCC")
    ax.xaxis.grid(False)
    sns.despine()

    # discard outliers
    features.drop(outliers, inplace=True)

    sns.violinplot(x="Group", y=feature, hue="Population", data=features, split=True, ax=ax,
                   palette={"Children": '#0067B1', "Elders": '#C60E4F'})

    """
    # Add lines between groups, when they are independent (p-value < 0.05)
    for population, group1, group2, pvalue in p_values:
        if pvalue < 0.05:
            y_max = max(features.loc[(features["Group"] == group1) & (features["Population"] == population), FEATURES[0]].max(),
                        features.loc[(features["Group"] == group2) & (features["Population"] == population), FEATURES[0]].max())
            x1 = groups.tolist().index(group1)
            x2 = groups.tolist().index(group2)
            ax.plot([x1, x2], [y_max, y_max], color='k')
            ax.text((x1+x2)*.5, y_max, "*", ha='center')
    """

    # curate legend
    handles, labels = ax.get_legend_handles_labels()
    labels = ["ABCD (Elders)", "P (Children)", ]
    ax.legend(handles, labels, loc="lower left")
    # no legend
    #ax.legend_.remove()

    # X axis labels
    mmse_labels = [f"MMSE {start}-{end}" for start, end in mmse_classes]
    age_labels = [f"Age {start}-{end}" for start, end in age_classes]
    ax.set_xticklabels([mmse + '\n' + age for mmse, age in zip(mmse_labels, age_labels)])

    # remove x-axis title
    ax.set_xlabel("")

    # curate y-axis title
    ax.set_ylabel(curate_feature_names([feature])[0])

    # Show plot
    plt.show()
    #plt.savefig(
    #    "/Users/saraiva/Desktop/Doktorand/Scientific Outputs/Journal Articles/RH-images/Distributions/" +
    #    "7_distribution_{}.png".format(feature), dpi=300, bbox_inches='tight')

