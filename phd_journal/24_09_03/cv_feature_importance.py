from os import mkdir
from os.path import exists
from pickle import dump

import imblearn
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from matplotlib import pyplot as plt
import seaborn as sns
from pandas import Series
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
import ImbalancedLearningRegression as iblr
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.svm import SVR

from read import *
from read import read_all_features
from utils import feature_wise_normalisation, get_classes, curate_feature_names


def sample_weight(y_true):
    targets_classes = np.round(y_true).astype(int)
    # Class frequencies
    unique_classes, class_counts = np.unique(targets_classes, return_counts=True)
    class_frequencies = dict(zip(unique_classes, class_counts))
    # Inverse of class frequencies
    class_weights = {cls: 1.0 / freq for cls, freq in class_frequencies.items()}
    # Assign weights to samples
    sample_weights = np.array([class_weights[cls] for cls in targets_classes])
    return sample_weights


def custom_cv(objects, targets, n_splits=5, random_state=42):
    """
    Custom Cross-Validation with Data Augmentation on-the-fly.

    Args:
        objects: A DataFrame of feature vectors
        targets: A Series of target values
        n_splits: Number of folds in CV

    Returns:
        The augmented training objects, test objects, training targets, and test targets.
    """

    # Stratify by bins of 5 years
    strata = (targets // 5).astype(int)
    #print("Strata:", strata)
    sss = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    #sss = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    # Select test_size % of examples in a random and stratified way
    split_res = sss.split(objects, strata)

    # for each fold...
    for train_indices, test_indices in split_res:
        print("Train:", len(train_indices))
        print("Test:", len(test_indices))

        # Make sets
        train_objets = objects.iloc[train_indices]
        train_targets = targets.iloc[train_indices]
        test_objects = objects.iloc[test_indices]
        test_targets = targets.iloc[test_indices]

        # Augment train set
        #print("Train examples before augmentation:", len(train_targets))
        #train_objets, train_targets = augment(train_objets, train_targets)
        #print("Train examples after augmentation:", len(train_targets))

        yield train_objets, test_objects, train_targets, test_targets


def cv(model, objects, targets, folds: int, random_state:int):

    all_important_features = {}

    for i, (train_objects, test_objects, train_targets, test_targets) in enumerate(custom_cv(objects, targets, n_splits=folds, random_state=random_state)):
        print(f"Fold {i+1}")

        # make sub-dir if not exists
        fold_path = join(out_path, str(i+1))
        if not exists(fold_path):
            mkdir(fold_path)

        # Train the model
        print(f"Train examples: {len(train_objects)}")
        model.fit(train_objects, train_targets)
        # save model
        with open(join(fold_path, "model.pkl"), 'wb') as f:
            dump(model, f)

        result = permutation_importance(model, test_objects, test_targets, scoring='r2', n_repeats=8, random_state=0, n_jobs=-1,
                                        sample_weight=sample_weight(test_targets))
        sorted_idx = result.importances_mean.argsort()
        # Get top 15 according to mean importance
        importances = result.importances_mean
        indices = np.argsort(importances)[::-1]
        top_15_features_importances = importances[indices[:10]]
        top_15_features_names = [FEATURES_SELECTED[i] for i in indices[:10]]
        np.savetxt(join(fold_path, "top_15_feature_importances_test.txt"), top_15_features_names, fmt='%s')

        # Curate feature names
        top_15_features_names = curate_feature_names(top_15_features_names)

        # Sum the importance value
        for i, f in enumerate(top_15_features_names):
            if f in all_important_features:
                all_important_features[f] += top_15_features_importances[i]
            else:
                all_important_features[f] = top_15_features_importances[i]

        # Bar plot of top 15 features with seaborn, with y-axis on the right side
        plt.rcParams['font.family'] = 'Arial'
        ax = sns.barplot(x=top_15_features_importances, y=top_15_features_names, color='#C60E4F')
        plt.xlabel('Feature Importance', fontsize=11)
        ax.yaxis.tick_right()
        ax.yaxis.set_label_position("right")
        ax.invert_xaxis()
        plt.yticks(fontsize=11)
        plt.xticks(fontsize=11)
        sns.despine(right=False, left=True)
        plt.savefig(join(fold_path, f"feature_importances_test.png"), dpi=300, bbox_inches='tight')

    # Sort of all important features and keep the top 10
    all_important_features = {k: v for k, v in sorted(all_important_features.items(), key=lambda item: item[1], reverse=True)}
    top_10_features = list(all_important_features.keys())[:15]
    np.savetxt(join(out_path, "top_all_feature_importances.txt"), top_10_features, fmt='%s')
    # Print the names and scores of the top 10 features
    print("Top 15 features:")
    for k, v in all_important_features.items():
        print(f"{k}: {v:.4f}")




out_path = './scheme10/cv'

# KJPP RFE (30) NEW!
FEATURES_SELECTED = ['COH#Frontal(R)-Temporal(L)#alpha', 'Hjorth#Mobility#F4', 'Spectral#RelativePower#C4#delta', 'Spectral#Flatness#Pz#gamma', 'Spectral#Flatness#P3#gamma', 'Spectral#RelativePower#Fpz#delta', 'Spectral#RelativePower#C3#beta', 'Spectral#Flatness#T4#beta', 'Spectral#RelativePower#C4#beta', 'COH#Frontal(R)-Occipital(R)#alpha', 'Spectral#RelativePower#Fz#beta', 'Spectral#RelativePower#Cz#beta', 'COH#Frontal(L)-Occipital(L)#beta', 'Hjorth#Mobility#Fz', 'Spectral#Diff#F7#gamma', 'Spectral#RelativePower#C3#delta', 'Hjorth#Mobility#C3', 'Spectral#RelativePower#Fz#delta', 'Hjorth#Mobility#C4', 'Spectral#EdgeFrequency#P3#alpha', 'Spectral#Flatness#P4#gamma', 'Hjorth#Complexity#C4', 'Spectral#EdgeFrequency#F8#gamma', 'COH#Frontal(R)-Parietal(L)#theta', 'Spectral#EdgeFrequency#T4#alpha', 'COH#Frontal(L)-Parietal(R)#gamma', 'Spectral#RelativePower#Cz#gamma', 'Hjorth#Mobility#Cz', 'COH#Frontal(L)-Parietal(R)#alpha', 'COH#Temporal(R)-Parietal(R)#theta']

# 1) Read features
features = read_all_features('KJPP', multiples=True)
features = features[FEATURES_SELECTED]
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

#Remove bad-diagnoses
BAD_DIAGNOSES = np.loadtxt("/Volumes/MMIS-Saraiv/Datasets/KJPP/session_ids/bad_diagnoses.txt", dtype=str)
GOOD_DIAGNOSES = features.index.difference(BAD_DIAGNOSES)
features_bad_diagnoses = features.drop(GOOD_DIAGNOSES, errors='ignore')
targets_bad_features = targets.drop(GOOD_DIAGNOSES, errors='ignore')
n_before = len(features)
features = features.drop(BAD_DIAGNOSES, errors='ignore')
targets = targets.drop(BAD_DIAGNOSES, errors='ignore')
print("Removed Bad diagnoses:", n_before - len(features))

# 4) Define model
model = SVR(kernel='rbf', C=10, epsilon=0.6)

# 5) Cross-Validation
cv(model, features, targets, 10,42)
