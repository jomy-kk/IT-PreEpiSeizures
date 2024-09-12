from os import mkdir
from os.path import exists
from pickle import dump

import imblearn
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from pandas import Series
from sklearn.ensemble import GradientBoostingRegressor
import ImbalancedLearningRegression as iblr
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.svm import SVR

from read import *
from read import read_all_features
from utils import feature_wise_normalisation, diagnoses_groups


def augment(features: pd.DataFrame, targets):
    targets = targets.round().astype(int)

    # Histogram before
    plt.hist(targets, bins=17, rwidth=0.8)
    plt.title("Before")
    plt.show()

    # 4.1. Data Augmentation for regression (target is continuous values)
    """
    features['target'] = targets
    # make indexes sequential numbers
    features.reset_index()
    features, targets = iblr.adasyn(features, 'target', k=3)
    targets = features['target']
    features = features.drop('target', axis=1)
    """

    # 4.2. Data Augmentation method = SMOTE-C
    """
    for k in (5, 4, 3, 2, 1):
        try:
            smote = SMOTE(random_state=42, k_neighbors=k, sampling_strategy='auto')
            features, targets = smote.fit_resample(features, targets)
            print(f"Worked with k={k}")
            break
        except ValueError as e:
            print(f"Did not work with k={k}:", e)
    """

    # Histogram after
    plt.hist(targets, bins=17, rwidth=0.8)
    plt.title("After")
    plt.show()

    # Normalisation after DA
    #features = feature_wise_normalisation(features, method='min-max')
    print("Features shape after DA:", features.shape)

    return features, targets


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

    r2_scores = []
    mse_scores = []
    mae_scores = []

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

        # Make regression plot of train set
        plt.figure(figsize=(6, 5))
        plt.rcParams['font.family'] = 'Arial'
        sns.regplot(x=train_targets, y=predictions, scatter_kws={'alpha': 0.3, 'color': '#C60E4F'},
                    line_kws={'color': '#C60E4F'})
        plt.xlabel('True Age (years)', fontsize=12)
        plt.ylabel('Predicted Age (years)', fontsize=12)
        plt.xlim(2, 24)
        plt.ylim(2, 24)
        plt.xticks([3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23], fontsize=11)
        plt.yticks([3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23], fontsize=11)
        plt.grid(linestyle='--', alpha=0.4)
        plt.box(False)
        plt.tight_layout()
        plt.savefig(join(fold_path, 'train.png'))

        # Test the model
        print(f"Test examples: {len(test_objects)}")
        predictions = model.predict(test_objects)
        # save Dataframe predictions | targets of test set
        res = pd.DataFrame({'predictions': predictions, 'targets': test_targets})
        res.to_csv(join(fold_path, 'predictions_targets.csv'))

        # Calculate the scores
        r2 = r2_score(test_targets, predictions)
        print(f"R2: {r2}")
        mse = mean_squared_error(test_targets, predictions)
        print(f"MSE: {mse}")
        mae = mean_absolute_error(test_targets, predictions)
        print(f"MAE: {mae}")

        # Append the scores
        r2_scores.append(r2)
        mse_scores.append(mse)
        mae_scores.append(mae)

        colors = get_colors(test_objects.index)

        # Make regression plot of test set
        plt.figure(figsize=(6, 5))
        plt.rcParams['font.family'] = 'Arial'
        sns.regplot(x=test_targets, y=predictions, scatter_kws={'alpha': 0.3, 'color': '#C60E4F'},
                    line_kws={'color': '#C60E4F'}, color=colors)
        plt.xlabel('True Age (years)', fontsize=12)
        plt.ylabel('Predicted Age (years)', fontsize=12)
        plt.xlim(2, 24)
        plt.ylim(2, 24)
        plt.xticks([3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23], fontsize=11)
        plt.yticks([3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23], fontsize=11)
        plt.grid(linestyle='--', alpha=0.4)
        plt.box(False)
        plt.tight_layout()
        plt.savefig(join(fold_path, 'test.png'))

    # Print the average scores
    print(f'Average R2: {np.mean(r2_scores)} +/- {np.std(r2_scores)}')
    print(f'Average MSE: {np.mean(mse_scores)} +/- {np.std(mse_scores)}')
    print(f'Average MAE: {np.mean(mae_scores)} +/- {np.std(mae_scores)}')


def get_colors(subject_codes):
    # make a colour for each diagnoses supergroup
    diagnoses_supergroups = ((0, 15), (15, 25), (25, 37), (37, 41))
    diagnoses_supergroups_colors = ['#FF0000', '#00FF00', '#0000FF', '#FFFF00']

    # Read diagnoses
    diagnoses = read_diagnoses('KJPP')
    colors = []
    for code in subject_codes:
        code = code.split('$')[0]
        subject_diagnoses = diagnoses[code]
        for d in subject_diagnoses:
            for i, D in enumerate(diagnoses_groups.values()):
                if d in D:
                    colors.append(diagnoses_supergroups_colors[i])
                    break

    return colors


out_path = './scheme1/cv'

#FEATURES_SELECTED = ['Spectral#Flatness#C3#alpha', 'COH#Occipital(L)-Occipital(R)#theta', 'Spectral#Entropy#Fpz#delta', 'Hjorth#Complexity#O2', 'PLI#Occipital(L)-Occipital(R)#theta', 'PLI#Temporal(R)-Occipital(R)#delta', 'Spectral#Entropy#T6#alpha', 'COH#Frontal(R)-Parietal(L)#beta', 'Spectral#Flatness#Fpz#delta', 'Spectral#Diff#P4#delta', 'Spectral#Diff#Fp1#gamma', 'Spectral#RelativePower#Cz#delta', 'Spectral#Diff#Fz#theta', 'Spectral#RelativePower#T5#theta', 'Hjorth#Mobility#F7', 'Hjorth#Complexity#P3', 'Spectral#Entropy#F8#delta', 'PLI#Temporal(L)-Parietal(R)#theta', 'Spectral#RelativePower#F4#gamma', 'PLI#Temporal(R)-Occipital(L)#delta', 'Hjorth#Activity#Cz', 'Spectral#Entropy#C4#theta', 'Spectral#EdgeFrequency#Fp1#beta', 'Spectral#EdgeFrequency#Pz#gamma', 'Spectral#RelativePower#P3#alpha', 'Spectral#Flatness#Cz#delta', 'COH#Parietal(L)-Parietal(R)#theta', 'Spectral#Entropy#Cz#gamma', 'Spectral#EdgeFrequency#Cz#alpha', 'Spectral#RelativePower#P3#beta']
FEATURES_SELECTED = ['Spectral#Diff#C3#theta', 'Spectral#RelativePower#C3#gamma', 'Spectral#RelativePower#C4#delta', 'Spectral#PeakFrequency#C4#beta', 'Spectral#Flatness#C4#gamma', 'Spectral#RelativePower#Cz#beta', 'Spectral#RelativePower#Cz#beta', 'Spectral#RelativePower#Cz#gamma', 'Spectral#Diff#Cz#gamma', 'Spectral#Entropy#F4#beta', 'Spectral#Flatness#F4#beta', 'Spectral#Entropy#F7#theta', 'Spectral#PeakFrequency#Fp2#alpha', 'Spectral#PeakFrequency#Fp2#beta', 'Spectral#RelativePower#Fz#delta', 'Spectral#PeakFrequency#Fz#theta', 'Spectral#PeakFrequency#Fz#gamma', 'Spectral#PeakFrequency#O1#beta', 'Spectral#Entropy#O2#delta', 'Spectral#PeakFrequency#O2#theta', 'Spectral#PeakFrequency#P3#gamma', 'Spectral#Diff#P4#beta', 'Spectral#EdgeFrequency#Pz#gamma', 'Spectral#EdgeFrequency#T3#delta', 'Spectral#Flatness#T5#alpha', 'Hjorth#Activity#C3', 'Hjorth#Activity#P4', 'Hjorth#Mobility#Cz', 'PLI#Temporal(L)-Parietal(L)#alpha', 'PLI#Temporal(L)-Occipital(L)#beta']
#FEATURES_SELECTED = ['Spectral#PeakFrequency#Fz#theta', 'Spectral#Entropy#T6#delta', 'Spectral#Diff#Cz#alpha', 'Spectral#Entropy#P3#gamma', 'Spectral#EdgeFrequency#T6#alpha', 'Hjorth#Activity#Fp1', 'Spectral#RelativePower#O2#beta', 'Spectral#Entropy#C4#delta', 'Spectral#Entropy#F3#alpha', 'Spectral#Flatness#Fp1#gamma', 'Spectral#Flatness#F8#beta', 'Spectral#Entropy#C4#gamma', 'PLI#Frontal(L)-Frontal(R)#gamma', 'Spectral#PeakFrequency#F8#beta', 'COH#Temporal(L)-Temporal(R)#alpha', 'PLI#Parietal(R)-Occipital(L)#delta', 'Spectral#PeakFrequency#T3#delta', 'Spectral#Entropy#F7#gamma', 'Spectral#Flatness#Fp2#gamma', 'Spectral#EdgeFrequency#F4#beta', 'Spectral#EdgeFrequency#T3#delta', 'Spectral#Flatness#P4#delta', 'Spectral#Diff#Fpz#beta', 'Spectral#Entropy#T3#gamma', 'COH#Parietal(L)-Occipital(R)#delta', 'COH#Temporal(R)-Occipital(L)#gamma', 'PLI#Temporal(R)-Occipital(L)#gamma', 'Spectral#PeakFrequency#O2#delta', 'Hjorth#Mobility#P4', 'Spectral#Diff#Fz#theta', 'Spectral#PeakFrequency#F4#beta', 'Spectral#Entropy#P3#beta', 'Spectral#Flatness#C3#theta', 'Spectral#EdgeFrequency#T3#theta', 'Spectral#PeakFrequency#Fp2#beta', 'PLI#Frontal(R)-Temporal(R)#theta', 'PLI#Frontal(L)-Parietal(R)#gamma', 'COH#Frontal(L)-Parietal(R)#beta', 'Hjorth#Complexity#P3', 'Spectral#Flatness#T5#theta', 'Spectral#Flatness#P4#beta', 'Spectral#RelativePower#O1#delta', 'Spectral#RelativePower#O2#alpha', 'COH#Frontal(L)-Parietal(R)#theta', 'Spectral#EdgeFrequency#Pz#delta', 'PLI#Frontal(R)-Occipital(L)#delta', 'Spectral#Entropy#F8#delta', 'Spectral#PeakFrequency#T4#alpha', 'PLI#Temporal(L)-Parietal(R)#theta', 'Spectral#RelativePower#T4#alpha', 'PLI#Frontal(R)-Occipital(R)#alpha', 'PLI#Temporal(R)-Occipital(R)#theta', 'Spectral#Diff#C3#delta', 'Spectral#Entropy#O2#gamma', 'Spectral#RelativePower#F8#gamma', 'COH#Temporal(L)-Parietal(L)#gamma', 'PLI#Frontal(R)-Occipital(R)#delta', 'Spectral#Flatness#Pz#theta', 'PLI#Frontal(L)-Parietal(R)#beta', 'Spectral#EdgeFrequency#T6#gamma', 'Spectral#PeakFrequency#C4#theta', 'Hjorth#Activity#Fp2', 'Spectral#EdgeFrequency#Fp1#beta', 'Spectral#EdgeFrequency#Pz#gamma', 'COH#Temporal(R)-Parietal(L)#alpha', 'Spectral#PeakFrequency#Fz#alpha', 'Spectral#Entropy#Cz#beta', 'Hjorth#Complexity#Fpz', 'COH#Temporal(R)-Parietal(R)#gamma', 'COH#Parietal(L)-Parietal(R)#theta', 'Spectral#Entropy#Fpz#alpha', 'Spectral#RelativePower#Pz#theta', 'COH#Parietal(R)-Occipital(L)#alpha', 'Spectral#PeakFrequency#T4#gamma', 'Spectral#Entropy#Cz#gamma', 'Spectral#EdgeFrequency#Cz#alpha', 'Spectral#Entropy#P3#alpha', 'COH#Parietal(R)-Occipital(L)#theta', 'Spectral#Entropy#Fz#beta', 'Spectral#RelativePower#P3#beta']

ixs_to_remove = [f for i, f in enumerate(FEATURES_SELECTED) if 'EdgeFrequency' in f]
for f in FEATURES_SELECTED:
    FEATURES_SELECTED.remove(f)

# 1) Read features
#hbn = read_all_features('Healthy Brain Network', multiples=True)
features = read_all_features('KJPP', multiples=True)
#features = pd.concat([hbn, kjpp], axis=0)
features = features[FEATURES_SELECTED]
print("Features Shape before drop subjects:", features.shape)
features = features.dropna(axis=0)
print("Features Shape after drop subjects:", features.shape)

# 2) Read targets
hbn_ages = read_ages('HBN')
kjpp_ages = read_ages('KJPP')
targets = Series()
for index in features.index:
    if '$' in str(index):  # Multiples
        key = str(index).split('$')[0]  # remove the multiple
    else:  # Original
        key = index

    if key in hbn_ages:
        targets.loc[index] = hbn_ages[key]
    if key in kjpp_ages:
        targets.loc[index] = kjpp_ages[key]

targets = targets.dropna()  # Drop subject_sessions with nans targets
features = features.loc[targets.index]
print("Features Shape before drop wo/ages:", features.shape)

# keep only ages <= 23
targets = targets[targets <= 18]
features = features.loc[targets.index]

# 3) Normalisation feature-wise
features = feature_wise_normalisation(features, method='min-max')

# 4) Define model
model = GradientBoostingRegressor(n_estimators=500, max_depth=15, random_state=0, loss='absolute_error', learning_rate=0.04, )
#model = SVR(degree=6, kernel='rbf', C=3)

# 5) Cross-Validation
cv(model, features, targets, 5,42)
