# We will compare the distribution of 6 features in two populations.
# Each population will have 3 groups.
# At the end we want violin plots with seaborn.

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from pandas import Series

from utils import curate_feature_names, feature_wise_normalisation
from read import read_all_features, read_ages

from scipy.stats import f_oneway, ks_2samp


def read_children():
    # 1) Read features
    features = read_all_features('KJPP', multiples=True)
    features = features.dropna()  # drop sessions with missing values
    features.index = features.index.str.split('$').str[0]  # remove $ from the index

    # 2) Read targets
    kjpp_ages = read_ages('KJPP')
    targets = Series()
    for index in features.index:
        if '$' in str(index):  # Multiples
            key = str(index).split('$')[0]  # remove the multiple
        else:  # Original
            key = index
        if key in kjpp_ages:
            targets.loc[index] = kjpp_ages[key]

    targets = targets.dropna()  # Drop subject_sessions with nans targets
    features = features.loc[targets.index]
    print("Features Shape before drop wo/ages:", features.shape)

    # keep only ages <= 23
    targets = targets[targets <= 23]
    features = features.loc[targets.index]

    # 3) Normalisation feature-wise
    features = feature_wise_normalisation(features, method='min-max')

    # Remove bad-diagnoses
    BAD_DIAGNOSES = np.loadtxt("/Volumes/MMIS-Saraiv/Datasets/KJPP/session_ids/bad_diagnoses.txt", dtype=str)
    GOOD_DIAGNOSES = features.index.difference(BAD_DIAGNOSES)
    n_before = len(features)
    features = features.drop(BAD_DIAGNOSES, errors='ignore')
    targets = targets.drop(BAD_DIAGNOSES, errors='ignore')
    print("Removed Bad diagnoses:", n_before - len(features))

    return features, targets


# Get elders and children data
features_children, targets_children = read_children()

# Groups with targets boundaries
age_classes = ((0, 8), (8, 10), (10, 12), (12,14), (14, 16), (16, 23))

# Assign groups and population
features_children["Group"] = np.nan

for i, (start, end) in enumerate(age_classes):
    features_children.loc[(targets_children >= start) & (targets_children <= end), "Group"] = i

# Drop NaN values
features_children.dropna(inplace=True)

# Features to compare

FEATURES = [
    "COH#Frontal(R)-Parietal(L)#theta",
    "COH#Frontal(L)-Parietal(R)#alpha",
    "COH#Frontal(L)-Occipital(L)#beta"
]

# Select only important features
features_children_ = features_children.loc[:, FEATURES + ["Group"]]
# Concatenate data
all_features = features_children_
all_features.reset_index(drop=True, inplace=True)

# Get the groups
groups = all_features["Group"].unique()

for feature in FEATURES:
    print("=====================================")
    print("Feature: {}".format(feature))

    # Get feature of interest
    features = all_features.loc[:, [feature, "Group"]]

    # Remove outliers
    outliers = []
    for group1 in groups:
        for group2 in groups:
            if group1 == group2:
                continue
            else:
                # Get the data
                data1 = features.loc[(features["Group"] == group1), feature]
                data2 = features.loc[(features["Group"] == group2), feature]
                # Select 10% by 1.5 standard deviations
                outliers.extend(data1[(data1 < data1.mean() - 1.5*data1.std()) | (data1 > data1.mean() + 1.5*data1.std())].index)
                outliers.extend(data2[(data2 < data2.mean() - 1.5*data2.std()) | (data2 > data2.mean() + 1.5*data2.std())].index)

    # Remove outliers
    outliers = list(set(outliers))
    features.drop(outliers, inplace=True)

    # Make violins plot
    # in the x-axis we will have 3 groups, each with two classes, elders and children, so 6 violins
    # in the y-axis we will have the feature values
    plt.rcParams['font.family'] = 'Arial'
    plt.rcParams['font.size'] = 12
    fig, ax = plt.figure(figsize=(16, 5)), plt.gca()
    # make grid only horizontal
    plt.grid(True, which='major', linestyle='--', linewidth=0.5, color="#BBBBBB")
    ax.xaxis.grid(False)
    sns.despine()

    sns.violinplot(x="Group", y=feature, data=features, split=True, ax=ax, color='#C60E4F', width=0.4)

    # curate legend
    handles, labels = ax.get_legend_handles_labels()
    labels = ["ABCD (Elders)", "P (Children)", ]
    ax.legend(handles, labels, loc="best")
    # no legend
    ax.legend_.remove()

    # X axis labels
    #age_labels = [f"{start} - {end}" for start, end in age_classes]
    #ax.set_xticklabels(age_labels)

    # no x-axis
    ax.set_xticks([])

    # remove x-axis title
    ax.set_xlabel("")

    # curate y-axis title
    #ax.set_ylabel(curate_feature_names([feature])[0])
    ax.set_ylabel("Normalised feature")

    # Show plot
    #plt.show()

    # plot area transparent
    ax.patch.set_alpha(0.0)

    # save with background transparent
    outpath = "./topomaps/" + f"distribution_{feature}.png"
    plt.savefig(outpath, dpi=300, bbox_inches='tight', transparent=True)

