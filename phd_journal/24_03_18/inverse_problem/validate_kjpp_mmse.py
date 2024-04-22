import numpy as np
from math import ceil, floor
from matplotlib import pyplot as plt
from pandas import Series
import seaborn as sns

from read import *
from utils import *

# FIXME
# kjpp + eldersly features selected (80)
FEATURES_SELECTED = ['Spectral#RelativePower#F3#gamma', 'Hjorth#Complexity#T3', 'Spectral#PeakFrequency#O1#beta3', 'Spectral#Entropy#Pz#beta1', 'Spectral#RelativePower#Cz#beta2', 'Spectral#Diff#P4#beta2', 'Spectral#Flatness#T5#alpha2', 'Spectral#PeakFrequency#Fz#beta3', 'Spectral#EdgeFrequency#T3#delta', 'PLI#Temporal(L)-Occipital(L)#beta1', 'Spectral#RelativePower#C4#delta', 'Spectral#PeakFrequency#F8#alpha1', 'Spectral#EdgeFrequency#Pz#gamma', 'Spectral#PeakFrequency#Cz#gamma', 'Spectral#Flatness#T6#gamma', 'Spectral#RelativePower#Fz#delta', 'Spectral#EdgeFrequency#Fz#beta3', 'Spectral#EdgeFrequency#F8#beta3', 'Spectral#Diff#Cz#gamma', 'Hjorth#Activity#C3', 'Spectral#RelativePower#Cz#delta', 'Spectral#RelativePower#Fp2#gamma', 'Spectral#Entropy#F7#theta', 'PLI#Temporal(L)-Parietal(L)#alpha2', 'Spectral#RelativePower#T4#beta1', 'Spectral#RelativePower#Cz#gamma', 'Hjorth#Activity#P4', 'Spectral#RelativePower#Fz#gamma', 'Spectral#RelativePower#P3#theta', 'Spectral#EdgeFrequency#O2#beta2', 'Spectral#Diff#C4#beta1', 'Spectral#RelativePower#C3#gamma', 'Spectral#RelativePower#P4#beta3', 'Spectral#PeakFrequency#Fp2#beta2', 'Spectral#EdgeFrequency#T3#theta', 'Spectral#RelativePower#Fp1#beta1', 'Hjorth#Mobility#Pz', 'Spectral#RelativePower#Fpz#gamma', 'Spectral#Diff#T4#beta1', 'Spectral#Entropy#P3#alpha1', 'Spectral#Flatness#F4#beta2', 'Spectral#Entropy#F4#beta2', 'Spectral#RelativePower#C4#gamma', 'Spectral#RelativePower#Cz#beta3', 'Spectral#RelativePower#O1#alpha2', 'Spectral#PeakFrequency#Fz#gamma', 'Spectral#PeakFrequency#F4#delta', 'Spectral#RelativePower#P4#alpha1', 'Spectral#PeakFrequency#P3#gamma', 'Spectral#RelativePower#O2#beta3', 'Hjorth#Mobility#P4', 'Hjorth#Complexity#Fp2', 'Spectral#Diff#P3#beta1', 'Spectral#RelativePower#C3#beta1', 'Spectral#EdgeFrequency#Cz#beta3', 'Spectral#Diff#C3#theta', 'Spectral#RelativePower#Fp1#beta2', 'Spectral#EdgeFrequency#F4#delta', 'Spectral#RelativePower#P3#beta3', 'Spectral#RelativePower#F3#theta', 'Spectral#Entropy#O2#delta', 'Spectral#PeakFrequency#C4#beta1', 'Spectral#EdgeFrequency#Cz#gamma', 'Spectral#RelativePower#T6#beta1', 'Spectral#PeakFrequency#O2#theta', 'Spectral#Flatness#C4#gamma', 'Spectral#PeakFrequency#F7#gamma', 'Spectral#RelativePower#F7#gamma', 'Spectral#Diff#T5#beta1', 'Spectral#EdgeFrequency#Pz#beta3', 'Spectral#RelativePower#T3#beta3', 'Spectral#RelativePower#T4#gamma', 'Spectral#EdgeFrequency#F3#theta', 'Spectral#RelativePower#Fp2#beta1', 'Spectral#PeakFrequency#Fz#theta', 'Hjorth#Complexity#T5', 'Hjorth#Complexity#T4', 'Hjorth#Complexity#F8', 'Hjorth#Mobility#Cz', 'Spectral#PeakFrequency#Fp2#alpha1']

# 1) Get all features
features = read_all_features('KJPP')

# 1.1) Select features
features = features[FEATURES_SELECTED]
print("Number of features selected:", len(features.columns))

# 1.2) Remove disorders
# FIXME
print("Number of subjects before removing disorders:", len(features))


# 2) Get targerts
targets = Series()
ages = read_ages('KJPP')
for session in features.index:
    age = ages[session]
    targets.loc[session] = age


# 3) Normalise features
"""
# 3.1) Calibrate features of adults (Age >= 18) to have the same mean and standard deviation as the elderly with MMSE == 30.
cal_ref = features[targets >= 18]
mmse30_stochastics = read_csv('elderly_mmse30_stochastic_pattern.csv', index_col=0)
for feature in cal_ref.columns:
    old_mean = cal_ref[feature].mean()
    old_std = cal_ref[feature].std()
    new_mean = mmse30_stochastics[feature]['mean']
    new_std = mmse30_stochastics[feature]['std']
    # transform
    cal_ref[feature] = (cal_ref[feature] - old_mean) * (new_std / old_std) + new_mean
# Understand the transformation done to reference and apply it to the remaining of the dataset
before = features[targets >= 18]
diff_mean = cal_ref.mean() - before.mean()
# Apply the difference to the rest of the dataset
cal_non_ref = features[targets < 18]
cal_non_ref = cal_non_ref + diff_mean
# Concatenate
features = pd.concat([cal_ref, cal_non_ref])
"""
# 2.2) Between 0 and 1
features = feature_wise_normalisation(features, 'min-max')


# 3) Convert features to an appropriate format
feature_names = features.columns.to_numpy()
sessions = features.index.to_numpy()
features = [features.loc[code].to_numpy() for code in sessions]


# 4) Load model
from pickle import load
with open('model_80feat.pkl', 'rb') as file:
    model = load(file)

# 5) Predict
predictions = model.predict(features)


def is_good_developmental_age_estimate(age: float, mmse: int, margin:float=0) -> bool:
    """
    Checks if the MMSE estimate is within the acceptable range for the given age.
    A margin can be added to the acceptable range.
    """
    assert 0 <= mmse <= 30, "MMSE must be between 0 and 30"
    assert 0 <= age, "Developmental age estimate must be positive"

    if age < 1.25:
        return 0 - margin <= mmse <= age / 2 + margin
    elif age < 2:
        return floor((4 * age / 15) - (1 / 3)) - margin <= mmse <= ceil(age / 2) + margin
    elif age < 5:
        return (4 * age / 15) - (1 / 3) - margin <= mmse <= 2 * age + 5 + margin
    elif age < 7:
        return 2 * age - 6 - margin <= mmse <= (4 * age / 3) + (25 / 3) + margin
    elif age < 8:
        return (4 * age / 5) + (47 / 5) - margin <= mmse <= (4 * age / 3) + (25 / 3) + margin
    elif age < 12:
        return (4 * age / 5) + (47 / 5) - margin <= mmse <= (4 * age / 5) + (68 / 5) + margin
    elif age < 13:
        return (4 * age / 7) + (92 / 7) - margin <= mmse <= (4 * age / 5) + (68 / 5) + margin
    elif age < 19:
        return (4 * age / 7) + (92 / 7) - margin<= mmse <= 30 + margin
    elif age >= 19:
        return mmse - margin >= 29 + margin


# 6) Plot
accurate = []
inaccurate = []
for prediction, age in zip(predictions, targets):
    if is_good_developmental_age_estimate(age, prediction, margin=1.5):
        accurate.append((age, prediction))
    else:
        inaccurate.append((age, prediction))

accurate_x, accurate_y = zip(*accurate)
inaccurate_x, inaccurate_y = zip(*inaccurate)

# 9. Plot predictions vs targets
plt.figure()
plt.ylabel('MMSE Estimate (units)')
plt.xlabel('Chronological Age (years)')
plt.xlim(2, 20)
plt.grid(linestyle='--', alpha=0.4)
plt.scatter(accurate_x, accurate_y, color='g', marker='.', alpha=0.3)
plt.scatter(inaccurate_x, inaccurate_y, color='r', marker='.', alpha=0.3)
# remove box around plot
plt.box(False)
plt.show()


# 10. Metrics

# Percentage right
percentage_right = len(accurate) / (len(accurate) + len(inaccurate))
print("Correct Bin Assignment:", percentage_right)

# R2 Score
from sklearn.metrics import r2_score
# Normalize between 0 and 1
targets_norm = (targets - targets.min()) / (targets.max() - targets.min())
predictions_norm = (predictions - predictions.min()) / (predictions.max() - predictions.min())
r2 = r2_score(targets_norm, predictions_norm)
print("R2 Score:", r2)

# pearson rank correlation
from scipy.stats import pearsonr
pearson, pvalue = pearsonr(targets, predictions)
print("Pearson rank correlation:", pearson, f"(p={pvalue})")

# Spearman rank correlation
from scipy.stats import spearmanr
spearman, pvalue = spearmanr(targets, predictions, alternative='greater')
print("Spearman rank correlation:", spearman, f"(p={pvalue})")

# Kendal rank correlation
from scipy.stats import kendalltau
kendall, pvalue = kendalltau(targets, predictions, alternative='greater')
print("Kendall rank correlation:", kendall, f"(p={pvalue})")

# Somers' D
from scipy.stats import somersd
res = somersd(targets, predictions)
correlation, pvalue, table = res.statistic, res.pvalue, res.table
print("Somers' D:", correlation, f"(p={pvalue})")

# Confusion Matrix
from sklearn.metrics import confusion_matrix
# We'll have 4 classes
# here are the boundaries
age_classes = ((0, 5), (5, 8), (8, 13), (13, 25))
mmse_classes = ((0, 9), (9, 15), (15, 24), (24, 30))

# assign predictions to classes
mmse_classes_assigned = []
for prediction in predictions:
    for i, (lower, upper) in enumerate(mmse_classes):
        if lower <= float(prediction) <= upper:
            mmse_classes_assigned.append(i)
            break
# assign targets to classes
age_classes_assigned = []
for age in targets:
    for i, (lower, upper) in enumerate(age_classes):
        if lower <= age <= upper:
            age_classes_assigned.append(i)
            break

# confusion matrix
conf_matrix = confusion_matrix(age_classes_assigned, mmse_classes_assigned)
# plot
plt.figure()
sns.heatmap(conf_matrix, annot=True, cmap='Blues', fmt='g')
plt.xlabel('True Chronological Age (years)')
plt.xticks([0, 1, 2, 3], ['0-5', '5-8', '8-13', '13-25'])
plt.ylabel('MMSE Estimate (units)')
plt.yticks([0, 1, 2, 3], ['0-9', '9-15', '15-24', '24-30'])
plt.show()






