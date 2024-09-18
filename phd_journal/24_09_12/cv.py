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
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
import ImbalancedLearningRegression as iblr
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.svm import SVR, SVC
import simple_icd_10 as icd

from read import *
from read import read_all_features
from utils import feature_wise_normalisation, diagnoses_groups, diagnoses_supergroups_colors, get_classes, custom_cv


def cv(model, objects, targets, folds: int, random_state:int):
    """
    My own Implementation of Cross-Validation with Data Augmentation on-the-fly that ensures that the same subject
    is not present in both training and test sets.
    Args:
        model: A Sklearn model
        objects: The feature vectors of each example in DataFrame format
        targets: The target values of each example in Series format
        folds: The number of folds, k
        stratified: Whether to use stratified k-fold
        shuffle: Whether to shuffle the data before splitting
        random_state: Random seed for reproducibility, if shuffle is True

    Prints:
        The average R2, MSE, and MAE scores across all folds.

    Plots:
        The regression plot between the true and predicted MMSE scores for each fold.
    """

    accuracy_scores = []
    f1_scores = []

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

        predictions = model.predict(train_objects)

        # Test the model
        print(f"Test examples: {len(test_objects)}")
        predictions = model.predict(test_objects)
        # save Dataframe predictions | targets of test set
        res = pd.DataFrame({'predictions': predictions, 'targets': test_targets})
        res.to_csv(join(fold_path, 'predictions_targets.csv'))

        # Calculate the weighted accuracy and f1 scores with sklearn
        class_weights = {diagnoses_supergroups_colors[i]: 1/len(diagnoses_supergroups_colors) for i in range(len(diagnoses_supergroups_colors))}
        sample_weights = [class_weights[t] for t in test_targets]
        accuracy = accuracy_score(test_targets, predictions, normalize=True, sample_weight=sample_weights)
        f1 = f1_score(test_targets, predictions, average='weighted')
        accuracy_scores.append(accuracy)
        f1_scores.append(f1)
        print("Accuracy:", accuracy)
        print("F1:", f1)

    # Print the average scores
    print("Average Accuracy:", np.mean(accuracy_scores))
    print("Average F1:", np.mean(f1_scores))



out_path = './scheme2/cv'

#FEATURES_SELECTED = ['Spectral#Flatness#C3#alpha', 'COH#Occipital(L)-Occipital(R)#theta', 'Spectral#Entropy#Fpz#delta', 'Hjorth#Complexity#O2', 'PLI#Occipital(L)-Occipital(R)#theta', 'PLI#Temporal(R)-Occipital(R)#delta', 'Spectral#Entropy#T6#alpha', 'COH#Frontal(R)-Parietal(L)#beta', 'Spectral#Flatness#Fpz#delta', 'Spectral#Diff#P4#delta', 'Spectral#Diff#Fp1#gamma', 'Spectral#RelativePower#Cz#delta', 'Spectral#Diff#Fz#theta', 'Spectral#RelativePower#T5#theta', 'Hjorth#Mobility#F7', 'Hjorth#Complexity#P3', 'Spectral#Entropy#F8#delta', 'PLI#Temporal(L)-Parietal(R)#theta', 'Spectral#RelativePower#F4#gamma', 'PLI#Temporal(R)-Occipital(L)#delta', 'Hjorth#Activity#Cz', 'Spectral#Entropy#C4#theta', 'Spectral#EdgeFrequency#Fp1#beta', 'Spectral#EdgeFrequency#Pz#gamma', 'Spectral#RelativePower#P3#alpha', 'Spectral#Flatness#Cz#delta', 'COH#Parietal(L)-Parietal(R)#theta', 'Spectral#Entropy#Cz#gamma', 'Spectral#EdgeFrequency#Cz#alpha', 'Spectral#RelativePower#P3#beta']
#FEATURES_SELECTED = ['Spectral#Diff#C3#theta', 'Spectral#RelativePower#C3#gamma', 'Spectral#RelativePower#C4#delta', 'Spectral#PeakFrequency#C4#beta', 'Spectral#Flatness#C4#gamma', 'Spectral#RelativePower#Cz#beta', 'Spectral#RelativePower#Cz#beta', 'Spectral#RelativePower#Cz#gamma', 'Spectral#Diff#Cz#gamma', 'Spectral#Entropy#F4#beta', 'Spectral#Flatness#F4#beta', 'Spectral#Entropy#F7#theta', 'Spectral#PeakFrequency#Fp2#alpha', 'Spectral#PeakFrequency#Fp2#beta', 'Spectral#RelativePower#Fz#delta', 'Spectral#PeakFrequency#Fz#theta', 'Spectral#PeakFrequency#Fz#gamma', 'Spectral#PeakFrequency#O1#beta', 'Spectral#Entropy#O2#delta', 'Spectral#PeakFrequency#O2#theta', 'Spectral#PeakFrequency#P3#gamma', 'Spectral#Diff#P4#beta', 'Spectral#EdgeFrequency#Pz#gamma', 'Spectral#EdgeFrequency#T3#delta', 'Spectral#Flatness#T5#alpha', 'Hjorth#Activity#C3', 'Hjorth#Activity#P4', 'Hjorth#Mobility#Cz', 'PLI#Temporal(L)-Parietal(L)#alpha', 'PLI#Temporal(L)-Occipital(L)#beta']
#FEATURES_SELECTED = ['Spectral#PeakFrequency#Fz#theta', 'Spectral#Entropy#T6#delta', 'Spectral#Diff#Cz#alpha', 'Spectral#Entropy#P3#gamma', 'Spectral#EdgeFrequency#T6#alpha', 'Hjorth#Activity#Fp1', 'Spectral#RelativePower#O2#beta', 'Spectral#Entropy#C4#delta', 'Spectral#Entropy#F3#alpha', 'Spectral#Flatness#Fp1#gamma', 'Spectral#Flatness#F8#beta', 'Spectral#Entropy#C4#gamma', 'PLI#Frontal(L)-Frontal(R)#gamma', 'Spectral#PeakFrequency#F8#beta', 'COH#Temporal(L)-Temporal(R)#alpha', 'PLI#Parietal(R)-Occipital(L)#delta', 'Spectral#PeakFrequency#T3#delta', 'Spectral#Entropy#F7#gamma', 'Spectral#Flatness#Fp2#gamma', 'Spectral#EdgeFrequency#F4#beta', 'Spectral#EdgeFrequency#T3#delta', 'Spectral#Flatness#P4#delta', 'Spectral#Diff#Fpz#beta', 'Spectral#Entropy#T3#gamma', 'COH#Parietal(L)-Occipital(R)#delta', 'COH#Temporal(R)-Occipital(L)#gamma', 'PLI#Temporal(R)-Occipital(L)#gamma', 'Spectral#PeakFrequency#O2#delta', 'Hjorth#Mobility#P4', 'Spectral#Diff#Fz#theta', 'Spectral#PeakFrequency#F4#beta', 'Spectral#Entropy#P3#beta', 'Spectral#Flatness#C3#theta', 'Spectral#EdgeFrequency#T3#theta', 'Spectral#PeakFrequency#Fp2#beta', 'PLI#Frontal(R)-Temporal(R)#theta', 'PLI#Frontal(L)-Parietal(R)#gamma', 'COH#Frontal(L)-Parietal(R)#beta', 'Hjorth#Complexity#P3', 'Spectral#Flatness#T5#theta', 'Spectral#Flatness#P4#beta', 'Spectral#RelativePower#O1#delta', 'Spectral#RelativePower#O2#alpha', 'COH#Frontal(L)-Parietal(R)#theta', 'Spectral#EdgeFrequency#Pz#delta', 'PLI#Frontal(R)-Occipital(L)#delta', 'Spectral#Entropy#F8#delta', 'Spectral#PeakFrequency#T4#alpha', 'PLI#Temporal(L)-Parietal(R)#theta', 'Spectral#RelativePower#T4#alpha', 'PLI#Frontal(R)-Occipital(R)#alpha', 'PLI#Temporal(R)-Occipital(R)#theta', 'Spectral#Diff#C3#delta', 'Spectral#Entropy#O2#gamma', 'Spectral#RelativePower#F8#gamma', 'COH#Temporal(L)-Parietal(L)#gamma', 'PLI#Frontal(R)-Occipital(R)#delta', 'Spectral#Flatness#Pz#theta', 'PLI#Frontal(L)-Parietal(R)#beta', 'Spectral#EdgeFrequency#T6#gamma', 'Spectral#PeakFrequency#C4#theta', 'Hjorth#Activity#Fp2', 'Spectral#EdgeFrequency#Fp1#beta', 'Spectral#EdgeFrequency#Pz#gamma', 'COH#Temporal(R)-Parietal(L)#alpha', 'Spectral#PeakFrequency#Fz#alpha', 'Spectral#Entropy#Cz#beta', 'Hjorth#Complexity#Fpz', 'COH#Temporal(R)-Parietal(R)#gamma', 'COH#Parietal(L)-Parietal(R)#theta', 'Spectral#Entropy#Fpz#alpha', 'Spectral#RelativePower#Pz#theta', 'COH#Parietal(R)-Occipital(L)#alpha', 'Spectral#PeakFrequency#T4#gamma', 'Spectral#Entropy#Cz#gamma', 'Spectral#EdgeFrequency#Cz#alpha', 'Spectral#Entropy#P3#alpha', 'COH#Parietal(R)-Occipital(L)#theta', 'Spectral#Entropy#Fz#beta', 'Spectral#RelativePower#P3#beta']
FEATURES_SELECTED = ['Spectral#PeakFrequency#P3#alpha', 'Spectral#Flatness#C3#alpha', 'Spectral#Entropy#F4#gamma', 'Spectral#PeakFrequency#Fz#theta', 'Spectral#Entropy#T3#theta', 'Spectral#Entropy#P3#gamma', 'COH#Occipital(L)-Occipital(R)#theta', 'COH#Temporal(L)-Occipital(R)#beta', 'Spectral#Entropy#F3#alpha', 'COH#Frontal(L)-Parietal(L)#beta', 'Spectral#Flatness#Fp1#gamma', 'COH#Frontal(R)-Temporal(L)#alpha', 'PLI#Frontal(L)-Frontal(R)#gamma', 'PLI#Occipital(L)-Occipital(R)#theta', 'PLI#Frontal(L)-Parietal(L)#gamma', 'Spectral#PeakFrequency#F8#beta', 'Spectral#PeakFrequency#T3#delta', 'Spectral#Flatness#Fp2#gamma', 'COH#Temporal(L)-Parietal(L)#delta', 'Spectral#Flatness#Fpz#theta', 'COH#Frontal(L)-Occipital(R)#theta', 'Spectral#Diff#P4#delta', 'Spectral#Diff#Fpz#beta', 'PLI#Temporal(L)-Occipital(L)#theta', 'COH#Temporal(R)-Occipital(R)#gamma', 'Spectral#EdgeFrequency#T4#delta', 'Spectral#Entropy#T3#gamma', 'Spectral#EdgeFrequency#T4#theta', 'PLI#Frontal(R)-Occipital(R)#theta', 'COH#Parietal(L)-Occipital(R)#delta', 'COH#Temporal(R)-Occipital(L)#gamma', 'PLI#Temporal(R)-Occipital(L)#gamma', 'Spectral#RelativePower#O1#alpha', 'Spectral#PeakFrequency#O2#delta', 'PLI#Parietal(L)-Occipital(L)#alpha', 'Hjorth#Mobility#T6', 'Spectral#PeakFrequency#F4#beta', 'Spectral#RelativePower#T5#theta', 'PLI#Frontal(R)-Temporal(R)#beta', 'Hjorth#Complexity#O2', 'COH#Parietal(L)-Occipital(R)#theta', 'Hjorth#Activity#Fz', 'Spectral#EdgeFrequency#O1#alpha', 'Spectral#PeakFrequency#C4#delta', 'PLI#Frontal(L)-Occipital(L)#delta', 'PLI#Frontal(L)-Parietal(R)#gamma', 'Spectral#PeakFrequency#Fz#delta', 'COH#Frontal(R)-Occipital(L)#alpha', 'Spectral#RelativePower#O1#delta', 'Spectral#PeakFrequency#F8#alpha', 'Spectral#RelativePower#T3#alpha', 'Spectral#EdgeFrequency#Pz#delta', 'COH#Frontal(R)-Temporal(L)#delta', 'Spectral#Flatness#P3#gamma', 'PLI#Frontal(R)-Occipital(R)#alpha', 'Spectral#RelativePower#F4#gamma', 'PLI#Temporal(R)-Occipital(R)#theta', 'PLI#Temporal(R)-Occipital(L)#delta', 'PLI#Parietal(L)-Occipital(R)#alpha', 'COH#Temporal(L)-Parietal(L)#gamma', 'PLI#Frontal(R)-Occipital(R)#delta', 'Spectral#PeakFrequency#P3#beta', 'Spectral#RelativePower#Fp2#alpha', 'Spectral#PeakFrequency#C4#theta', 'PLI#Parietal(R)-Occipital(R)#beta', 'Spectral#EdgeFrequency#Pz#gamma', 'COH#Temporal(R)-Parietal(L)#alpha', 'Spectral#Diff#T5#delta', 'Spectral#Diff#T3#theta', 'Spectral#PeakFrequency#Fz#alpha', 'Spectral#Entropy#Cz#beta', 'Spectral#PeakFrequency#Pz#theta', 'PLI#Parietal(L)-Occipital(L)#beta', 'Spectral#Entropy#Fpz#alpha', 'Spectral#RelativePower#Pz#theta', 'COH#Parietal(R)-Occipital(L)#alpha', 'Spectral#Entropy#F8#alpha', 'Spectral#PeakFrequency#T4#gamma', 'COH#Parietal(R)-Occipital(L)#theta', 'Spectral#Entropy#Fz#beta']

"""
ixs_to_remove = [f for i, f in enumerate(FEATURES_SELECTED) if 'EdgeFrequency' in f]
for f in FEATURES_SELECTED:
    FEATURES_SELECTED.remove(f)
"""

# 1) Read features
features = read_all_features('KJPP', multiples=True)
features = features[FEATURES_SELECTED]
print("Features Shape before drop subjects:", features.shape)
features = features.dropna(axis=0)
print("Features Shape after drop subjects:", features.shape)

# 2) Read targets
all_diagnoses = get_classes(features.index)
targets = Series(all_diagnoses, index=features.index)
targets = targets.dropna()  # Drop subject_sessions with None targets; "gray" color
features = features.loc[targets.index]
print("Features Shape before drop wo/ages:", features.shape)

# 3) Normalisation feature-wise
features = feature_wise_normalisation(features, method='min-max')

# 4) Define model
#model = GradientBoostingClassifier(n_estimators=300, max_depth=15, random_state=0, learning_rate=0.04)
#model = SVC(degree=6, kernel='rbf', C=3)
model = GradientBoostingClassifier(n_estimators=200, max_depth=10, random_state=0, learning_rate=0.04)

# 5) Cross-Validation
cv(model, features, targets, 5,42)
