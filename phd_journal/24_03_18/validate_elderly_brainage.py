from pandas import Series

from read import *
from utils import *

# FIXME
FEATURES_SELECTED = ['Spectral#Diff#C3#theta', 'Spectral#RelativePower#C3#gamma', 'Spectral#RelativePower#C4#delta', 'Spectral#PeakFrequency#C4#beta1', 'Spectral#Flatness#C4#gamma', 'Spectral#RelativePower#Cz#beta2', 'Spectral#RelativePower#Cz#beta3', 'Spectral#RelativePower#Cz#gamma', 'Spectral#Diff#Cz#gamma', 'Spectral#Entropy#F4#beta2', 'Spectral#Flatness#F4#beta2', 'Spectral#Entropy#F7#theta', 'Spectral#PeakFrequency#Fp2#alpha1', 'Spectral#PeakFrequency#Fp2#beta2', 'Spectral#RelativePower#Fz#delta', 'Spectral#PeakFrequency#Fz#theta', 'Spectral#PeakFrequency#Fz#gamma', 'Spectral#PeakFrequency#O1#beta3', 'Spectral#Entropy#O2#delta', 'Spectral#PeakFrequency#O2#theta', 'Spectral#PeakFrequency#P3#gamma', 'Spectral#Diff#P4#beta2', 'Spectral#EdgeFrequency#Pz#gamma', 'Spectral#EdgeFrequency#T3#delta', 'Spectral#Flatness#T5#alpha2', 'Hjorth#Activity#C3', 'Hjorth#Activity#P4', 'Hjorth#Mobility#Cz', 'PLI#Temporal(L)-Parietal(L)#alpha2', 'PLI#Temporal(L)-Occipital(L)#beta1']

# 1) Get all features
insight = read_all_features('INSIGHT')
insight = insight[FEATURES_SELECTED]
sapienza = read_all_features('Sapienza')
sapienza = sapienza[FEATURES_SELECTED]
elderly = pd.concat([insight, sapienza], axis=0)

# Drop subject_sessions with nans
elderly = elderly.dropna()

# 2) Normalise features

# 2.1) With mean-std of KJPP
elderly = feature_wise_normalisation_with_coeffs(elderly, 'mean-std', 'kjpp_stochastic_pattern.csv')

# 3) Get all targets
insight_brainages = read_brainage('INSIGHT')
insight_ages = read_ages('INSIGHT')
sapienza_brainages = read_brainage('Sapienza')
sapienza_ages = read_ages('Sapienza')
# Assign targets to elderly
targets = Series()
for index in elderly.index:
    if '_' in index:
        key = int(index.split('_')[0])
        if key in insight_brainages:
            targets.loc[index] = insight_brainages[key] - insight_ages[key]
    elif index in sapienza_brainages:
        key = index
        targets.loc[index] = sapienza_brainages[key] - sapienza_ages[key]
    else:
        print('No target for', index)

"""
# 2.2) Calibrate features with target > 1 making them have the same mean and std of KJPP adults
cal_ref = elderly[targets > 1]
adult_stochastics = read_csv('kjpp_adult_stochastic_pattern.csv', index_col=0)

for feature in cal_ref.columns:
    old_mean = cal_ref[feature].mean()
    old_std = cal_ref[feature].std()
    new_mean = adult_stochastics[feature]['mean']
    new_std = adult_stochastics[feature]['std']
    # transform
    cal_ref[feature] = (cal_ref[feature] - old_mean) * (new_std / old_std) + new_mean

# Understand the transformation done to reference and apply it to the remaining of the dataset
before = elderly[targets > 1]
diff_mean = cal_ref.mean() - before.mean()
diff_std = cal_ref.std() - before.std()

# Apply the difference to the rest of the dataset
cal_non_ref = elderly[targets <= 1]
cal_non_ref = cal_non_ref * (diff_std / cal_non_ref.std()) + diff_mean

# Concatenate
elderly = pd.concat([cal_ref, cal_non_ref])
"""

# 4) Load model
from pickle import load
with open('model.pkl', 'rb') as file:
    model = load(file)

# 5) Predict
predictions = model.predict(elderly)

# 6) Plot
from matplotlib import pyplot as plt
plt.scatter(targets, predictions, alpha=0.5)
plt.xlabel('MRI Brain Age - Chronological Age (ground truth)')
plt.ylabel('Predicted Developmental Age (years)')
plt.show()






