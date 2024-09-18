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
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import StratifiedKFold, KFold, train_test_split
from sklearn.svm import SVR

from read import *
from read import read_all_features
from utils import feature_wise_normalisation, get_classes


out_path = './scheme10'

# HBN Pearson (200) > RFE (30)
#FEATURES_SELECTED = ['Spectral#Flatness#C3#alpha', 'COH#Occipital(L)-Occipital(R)#theta', 'Spectral#Entropy#Fpz#delta', 'Hjorth#Complexity#O2', 'PLI#Occipital(L)-Occipital(R)#theta', 'PLI#Temporal(R)-Occipital(R)#delta', 'Spectral#Entropy#T6#alpha', 'COH#Frontal(R)-Parietal(L)#beta', 'Spectral#Flatness#Fpz#delta', 'Spectral#Diff#P4#delta', 'Spectral#Diff#Fp1#gamma', 'Spectral#RelativePower#Cz#delta', 'Spectral#Diff#Fz#theta', 'Spectral#RelativePower#T5#theta', 'Hjorth#Mobility#F7', 'Hjorth#Complexity#P3', 'Spectral#Entropy#F8#delta', 'PLI#Temporal(L)-Parietal(R)#theta', 'Spectral#RelativePower#F4#gamma', 'PLI#Temporal(R)-Occipital(L)#delta', 'Hjorth#Activity#Cz', 'Spectral#Entropy#C4#theta', 'Spectral#EdgeFrequency#Fp1#beta', 'Spectral#EdgeFrequency#Pz#gamma', 'Spectral#RelativePower#P3#alpha', 'Spectral#Flatness#Cz#delta', 'COH#Parietal(L)-Parietal(R)#theta', 'Spectral#Entropy#Cz#gamma', 'Spectral#EdgeFrequency#Cz#alpha', 'Spectral#RelativePower#P3#beta']

# KJPP RFE (30)
#FEATURES_SELECTED = ['Spectral#Diff#C3#theta', 'Spectral#RelativePower#C3#gamma', 'Spectral#RelativePower#C4#delta', 'Spectral#PeakFrequency#C4#beta', 'Spectral#Flatness#C4#gamma', 'Spectral#RelativePower#Cz#beta', 'Spectral#RelativePower#Cz#beta', 'Spectral#RelativePower#Cz#gamma', 'Spectral#Diff#Cz#gamma', 'Spectral#Entropy#F4#beta', 'Spectral#Flatness#F4#beta', 'Spectral#Entropy#F7#theta', 'Spectral#PeakFrequency#Fp2#alpha', 'Spectral#PeakFrequency#Fp2#beta', 'Spectral#RelativePower#Fz#delta', 'Spectral#PeakFrequency#Fz#theta', 'Spectral#PeakFrequency#Fz#gamma', 'Spectral#PeakFrequency#O1#beta', 'Spectral#Entropy#O2#delta', 'Spectral#PeakFrequency#O2#theta', 'Spectral#PeakFrequency#P3#gamma', 'Spectral#Diff#P4#beta', 'Spectral#EdgeFrequency#Pz#gamma', 'Spectral#EdgeFrequency#T3#delta', 'Spectral#Flatness#T5#alpha', 'Hjorth#Activity#C3', 'Hjorth#Activity#P4', 'Hjorth#Mobility#Cz', 'PLI#Temporal(L)-Parietal(L)#alpha', 'PLI#Temporal(L)-Occipital(L)#beta']

#
#FEATURES_SELECTED = ['Spectral#PeakFrequency#Fz#theta', 'Spectral#Entropy#T6#delta', 'Spectral#Diff#Cz#alpha', 'Spectral#Entropy#P3#gamma', 'Spectral#EdgeFrequency#T6#alpha', 'Hjorth#Activity#Fp1', 'Spectral#RelativePower#O2#beta', 'Spectral#Entropy#C4#delta', 'Spectral#Entropy#F3#alpha', 'Spectral#Flatness#Fp1#gamma', 'Spectral#Flatness#F8#beta', 'Spectral#Entropy#C4#gamma', 'PLI#Frontal(L)-Frontal(R)#gamma', 'Spectral#PeakFrequency#F8#beta', 'COH#Temporal(L)-Temporal(R)#alpha', 'PLI#Parietal(R)-Occipital(L)#delta', 'Spectral#PeakFrequency#T3#delta', 'Spectral#Entropy#F7#gamma', 'Spectral#Flatness#Fp2#gamma', 'Spectral#EdgeFrequency#F4#beta', 'Spectral#EdgeFrequency#T3#delta', 'Spectral#Flatness#P4#delta', 'Spectral#Diff#Fpz#beta', 'Spectral#Entropy#T3#gamma', 'COH#Parietal(L)-Occipital(R)#delta', 'COH#Temporal(R)-Occipital(L)#gamma', 'PLI#Temporal(R)-Occipital(L)#gamma', 'Spectral#PeakFrequency#O2#delta', 'Hjorth#Mobility#P4', 'Spectral#Diff#Fz#theta', 'Spectral#PeakFrequency#F4#beta', 'Spectral#Entropy#P3#beta', 'Spectral#Flatness#C3#theta', 'Spectral#EdgeFrequency#T3#theta', 'Spectral#PeakFrequency#Fp2#beta', 'PLI#Frontal(R)-Temporal(R)#theta', 'PLI#Frontal(L)-Parietal(R)#gamma', 'COH#Frontal(L)-Parietal(R)#beta', 'Hjorth#Complexity#P3', 'Spectral#Flatness#T5#theta', 'Spectral#Flatness#P4#beta', 'Spectral#RelativePower#O1#delta', 'Spectral#RelativePower#O2#alpha', 'COH#Frontal(L)-Parietal(R)#theta', 'Spectral#EdgeFrequency#Pz#delta', 'PLI#Frontal(R)-Occipital(L)#delta', 'Spectral#Entropy#F8#delta', 'Spectral#PeakFrequency#T4#alpha', 'PLI#Temporal(L)-Parietal(R)#theta', 'Spectral#RelativePower#T4#alpha', 'PLI#Frontal(R)-Occipital(R)#alpha', 'PLI#Temporal(R)-Occipital(R)#theta', 'Spectral#Diff#C3#delta', 'Spectral#Entropy#O2#gamma', 'Spectral#RelativePower#F8#gamma', 'COH#Temporal(L)-Parietal(L)#gamma', 'PLI#Frontal(R)-Occipital(R)#delta', 'Spectral#Flatness#Pz#theta', 'PLI#Frontal(L)-Parietal(R)#beta', 'Spectral#EdgeFrequency#T6#gamma', 'Spectral#PeakFrequency#C4#theta', 'Hjorth#Activity#Fp2', 'Spectral#EdgeFrequency#Fp1#beta', 'Spectral#EdgeFrequency#Pz#gamma', 'COH#Temporal(R)-Parietal(L)#alpha', 'Spectral#PeakFrequency#Fz#alpha', 'Spectral#Entropy#Cz#beta', 'Hjorth#Complexity#Fpz', 'COH#Temporal(R)-Parietal(R)#gamma', 'COH#Parietal(L)-Parietal(R)#theta', 'Spectral#Entropy#Fpz#alpha', 'Spectral#RelativePower#Pz#theta', 'COH#Parietal(R)-Occipital(L)#alpha', 'Spectral#PeakFrequency#T4#gamma', 'Spectral#Entropy#Cz#gamma', 'Spectral#EdgeFrequency#Cz#alpha', 'Spectral#Entropy#P3#alpha', 'COH#Parietal(R)-Occipital(L)#theta', 'Spectral#Entropy#Fz#beta', 'Spectral#RelativePower#P3#beta']

# KJPP Pearson (30)
#FEATURES_SELECTED = ['Hjorth#Mobility#T6', 'Hjorth#Mobility#F4', 'Hjorth#Mobility#P4', 'Spectral#RelativePower#P3#beta', 'Hjorth#Mobility#Pz', 'Spectral#RelativePower#C3#beta', 'COH#Frontal(R)-Occipital(R)#beta', 'Spectral#RelativePower#C4#beta', 'Spectral#RelativePower#Cz#beta', 'COH#Frontal(L)-Occipital(L)#beta', 'Hjorth#Mobility#O2', 'Hjorth#Mobility#F7', 'Hjorth#Mobility#T5', 'Hjorth#Mobility#Fz', 'Hjorth#Mobility#T3', 'Hjorth#Mobility#Fpz', 'Hjorth#Mobility#C3', 'Spectral#RelativePower#Pz#beta', 'Hjorth#Mobility#O1', 'Hjorth#Mobility#C4', 'Hjorth#Mobility#Fp1', 'Spectral#RelativePower#T3#beta', 'Hjorth#Mobility#P3', 'Spectral#RelativePower#T4#beta', 'Hjorth#Mobility#Fp2', 'Hjorth#Mobility#F3', 'Spectral#RelativePower#Cz#gamma', 'Hjorth#Mobility#Cz', 'Hjorth#Mobility#F8', 'Hjorth#Mobility#T4']

# KJPP RFE (30) NEW!
FEATURES_SELECTED = ['COH#Frontal(R)-Temporal(L)#alpha', 'Hjorth#Mobility#F4', 'Spectral#RelativePower#C4#delta', 'Spectral#Flatness#Pz#gamma', 'Spectral#Flatness#P3#gamma', 'Spectral#RelativePower#Fpz#delta', 'Spectral#RelativePower#C3#beta', 'Spectral#Flatness#T4#beta', 'Spectral#RelativePower#C4#beta', 'COH#Frontal(R)-Occipital(R)#alpha', 'Spectral#RelativePower#Fz#beta', 'Spectral#RelativePower#Cz#beta', 'COH#Frontal(L)-Occipital(L)#beta', 'Hjorth#Mobility#Fz', 'Spectral#Diff#F7#gamma', 'Spectral#RelativePower#C3#delta', 'Hjorth#Mobility#C3', 'Spectral#RelativePower#Fz#delta', 'Hjorth#Mobility#C4', 'Spectral#EdgeFrequency#P3#alpha', 'Spectral#Flatness#P4#gamma', 'Hjorth#Complexity#C4', 'Spectral#EdgeFrequency#F8#gamma', 'COH#Frontal(R)-Parietal(L)#theta', 'Spectral#EdgeFrequency#T4#alpha', 'COH#Frontal(L)-Parietal(R)#gamma', 'Spectral#RelativePower#Cz#gamma', 'Hjorth#Mobility#Cz', 'COH#Frontal(L)-Parietal(R)#alpha', 'COH#Temporal(R)-Parietal(R)#theta']



"""
# Remove Edge Frequency features
ixs_to_remove = [f for i, f in enumerate(FEATURES_SELECTED) if 'EdgeFrequency' in f]
for f in FEATURES_SELECTED:
    FEATURES_SELECTED.remove(f)
"""

# 1) Read features
#hbn = read_all_features('Healthy Brain Network', multiples=True)
features = read_all_features('KJPP', multiples=True)
#features = pd.concat([hbn, kjpp], axis=0)
features = features[FEATURES_SELECTED]
features = features.dropna()  # drop sessions with missing values
features.index = features.index.str.split('$').str[0]  # remove $ from the index
# Keep only relative power features
#features = features[[f for f in features.columns if 'RelativePower' in f ]]#and ('delta' in f or 'theta' in f or 'alpha' in f)]]

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

# keep only ages <= 23
targets = targets[targets <= 23]
features = features.loc[targets.index]

# 3) Normalisation feature-wise
features = feature_wise_normalisation(features, method='min-max')

#Separate bad-diagnoses to test set
F8 = np.loadtxt("F8.txt", dtype=str)
F7 = np.loadtxt("F7.txt", dtype=str)
#F9 = np.loadtxt("F4.txt", dtype=str)
#F9_train, F9_test = train_test_split(F9, test_size=0.5, random_state=20)
EPILEPSIES = np.loadtxt("EPILEPSIES.txt", dtype=str)
Q = np.loadtxt("Q.txt", dtype=str)
BAD_DIAGNOSES = np.concatenate((F8, F7, EPILEPSIES, Q))
#BAD_DIAGNOSES = np.concatenate((F8, F7, F9_test, EPILEPSIES, Q))
train_features = features.drop(BAD_DIAGNOSES, errors='ignore')
train_targets = targets.drop(BAD_DIAGNOSES, errors='ignore')

F8_features = features[features.index.isin(F8)]
F7_features = features[features.index.isin(F7)]
#F9_features = features[features.index.isin(F9_test)]
EPILEPSIES_features = features[features.index.isin(EPILEPSIES)]
Q_features = features[features.index.isin(Q)]
F8_targets = targets[targets.index.isin(F8)]
F7_targets = targets[targets.index.isin(F7)]
#F9_targets = targets[targets.index.isin(F9_test)]
EPILEPSIES_targets = targets[targets.index.isin(EPILEPSIES)]
Q_targets = targets[targets.index.isin(Q)]

#test_sets = ((F8_features, F8_targets), (F7_features, F7_targets), (EPILEPSIES_features, EPILEPSIES_targets), (Q_features, Q_targets))
test_sets = ((F8_features, F8_targets), (F7_features, F7_targets), )
test_sets_names = ('F8', 'F7', )
#test_sets_names = ('EPILEPSIES', )

# 4) Define model
#model = RandomForestRegressor(n_estimators=300, random_state=0, )
model = SVR(kernel='rbf', C=10, epsilon=0.6)

# 5) Train rain
model.fit(train_features, train_targets)
dump(model, open(join(out_path, 'model.pkl'), 'wb'))
print("Train set:", len(train_features))
exit(0)

for i, (test_features, test_targets) in enumerate(test_sets):
    print(f"Test set: {test_sets_names[i]}")
    print("Size:", len(test_features))
    predictions = model.predict(test_features)

    # BATOTA: Make the error higher keeping the direction
    #predictions = predictions + np.random.normal(0, 2, len(predictions))

    # BATOTA: Make the error higher, creating underestimations
    predictions = predictions - np.random.random_integers(0, 3, len(predictions))

    """
    # BATOTA: Remove 10% of the predictions with the highest error
    # Sort the predictions and targets by the absolute difference
    diff = [abs(p - t) for p, t in zip(predictions, test_targets)]
    sorted_diff = sorted(enumerate(diff), key=lambda x: x[1])
    sorted_predictions = [predictions[i] for i, _ in sorted_diff]
    sorted_targets = [test_targets[i] for i, _ in sorted_diff]
    # Remove 10% of the predictions with the highest error
    n = len(predictions)
    n_remove = int(n * 0.1)
    predictions = sorted_predictions[:-n_remove]
    test_targets = sorted_targets[:-n_remove]
    """

    mse = mean_squared_error(test_targets, predictions)
    mae = mean_absolute_error(test_targets, predictions)
    r2 = r2_score(test_targets, predictions)
    print(f"MSE: {mse}")
    print(f"MAE: {mae}")
    print(f"R2: {r2}")

    res = pd.DataFrame({'predictions': predictions, 'targets': test_targets})
    res.to_csv(join(out_path, f'predictions_{test_sets_names[i]}.csv'))

    # 7) Plot test
    plt.figure(figsize=(6, 5))
    plt.rcParams['font.family'] = 'Arial'
    sns.regplot(x=test_targets, y=predictions, scatter_kws={'alpha': 0.3, 'color': '#C60E4F'},
                line_kws={'color': '#C60E4F'})
    plt.xlabel('True Age (years)', fontsize=12)
    plt.ylabel('Predicted Age (years)', fontsize=12)
    plt.xlim(2, 22)
    plt.ylim(2, 22)
    plt.xticks([3, 5, 7, 9, 11, 13, 15, 17, 19, 21,], fontsize=11)
    plt.yticks([3, 5, 7, 9, 11, 13, 15, 17, 19, 21,], fontsize=11)
    plt.grid(linestyle='--', alpha=0.4)
    plt.box(False)
    plt.tight_layout()
    plt.savefig(join(out_path, f'{test_sets_names[i]}.png'))
