import numpy as np
from math import ceil, floor
from matplotlib import pyplot as plt
from pandas import Series
import seaborn as sns

from read import *
from utils import *

# FIXME
FEATURES_SELECTED = ['Spectral#RelativePower#C3#beta1', 'Spectral#EdgeFrequency#C3#beta3', 'Spectral#RelativePower#C3#gamma', 'Spectral#EdgeFrequency#C4#alpha1', 'Spectral#RelativePower#C4#beta3', 'Spectral#EdgeFrequency#C4#beta3', 'Spectral#EdgeFrequency#C4#gamma', 'Spectral#Flatness#Cz#theta', 'Spectral#PeakFrequency#Cz#theta', 'Spectral#EdgeFrequency#Cz#beta3', 'Spectral#EdgeFrequency#Cz#gamma', 'Spectral#PeakFrequency#Cz#gamma', 'Spectral#RelativePower#F3#beta1', 'Spectral#Diff#F4#delta', 'Spectral#RelativePower#F7#beta3', 'Spectral#EdgeFrequency#F7#beta3', 'Spectral#RelativePower#F7#gamma', 'Spectral#RelativePower#F8#beta1', 'Spectral#EdgeFrequency#F8#beta3', 'Spectral#RelativePower#Fp1#beta1', 'Spectral#EdgeFrequency#Fp1#beta3', 'Spectral#Diff#Fp2#delta', 'Spectral#RelativePower#Fp2#beta1', 'Spectral#RelativePower#Fp2#beta3', 'Spectral#Diff#Fpz#beta2', 'Spectral#Entropy#O1#delta', 'Spectral#RelativePower#O1#beta2', 'Spectral#EdgeFrequency#O1#beta2', 'Spectral#EdgeFrequency#O1#beta3', 'Spectral#RelativePower#O2#delta', 'Spectral#PeakFrequency#O2#alpha1', 'Spectral#RelativePower#O2#beta1', 'Spectral#RelativePower#O2#beta3', 'Spectral#Diff#P3#beta1', 'Spectral#RelativePower#P3#beta3', 'Spectral#RelativePower#Pz#alpha1', 'Spectral#EdgeFrequency#Pz#beta3', 'Spectral#RelativePower#T4#alpha1', 'Spectral#RelativePower#T4#beta3', 'Spectral#RelativePower#T4#gamma', 'Spectral#EdgeFrequency#T5#beta2', 'Hjorth#Complexity#T5', 'Hjorth#Complexity#P4', 'Hjorth#Complexity#F7', 'Hjorth#Complexity#T4', 'Hjorth#Complexity#F8', 'Hjorth#Complexity#T3', 'Hjorth#Mobility#P3', 'PLI#Frontal(L)-Temporal(R)#alpha1', 'PLI#Frontal(L)-Occipital(L)#alpha1', 'PLI#Frontal(R)-Temporal(R)#alpha1', 'PLI#Temporal(R)-Parietal(R)#alpha1', 'PLI#Temporal(R)-Occipital(L)#alpha1', 'PLI#Parietal(R)-Occipital(L)#alpha1', 'PLI#Occipital(L)-Occipital(R)#alpha1', 'PLI#Temporal(R)-Occipital(R)#alpha2', 'PLI#Parietal(R)-Occipital(L)#alpha2', 'COH#Frontal(L)-Frontal(R)#theta', 'COH#Frontal(L)-Occipital(L)#theta', 'COH#Frontal(L)-Occipital(R)#alpha1', 'COH#Frontal(R)-Occipital(L)#alpha1', 'COH#Parietal(R)-Occipital(L)#alpha1', 'COH#Frontal(L)-Frontal(R)#alpha2', 'COH#Frontal(L)-Occipital(R)#alpha2', 'COH#Parietal(R)-Occipital(L)#alpha2', 'COH#Parietal(R)-Occipital(R)#alpha2', 'COH#Occipital(L)-Occipital(R)#alpha2', 'COH#Frontal(L)-Occipital(L)#beta1', 'COH#Temporal(R)-Parietal(R)#beta1', 'COH#Parietal(R)-Occipital(R)#beta1', 'COH#Frontal(L)-Parietal(L)#beta2', 'COH#Frontal(R)-Occipital(L)#beta2', 'COH#Frontal(L)-Temporal(R)#beta3', 'COH#Frontal(L)-Parietal(L)#beta3', 'COH#Frontal(L)-Occipital(L)#beta3', 'COH#Frontal(L)-Occipital(R)#beta3', 'COH#Frontal(R)-Occipital(L)#beta3', 'COH#Temporal(L)-Occipital(R)#beta3', 'COH#Frontal(L)-Occipital(R)#gamma', 'COH#Frontal(R)-Occipital(R)#gamma']

# 1) Get all features
features = read_all_features('KJPP')

# 1.1) Select features
features = features[FEATURES_SELECTED]
print("Number of features selected:", len(features.columns))

# Drop sessions with missing values
features = features.dropna()

# 2) Get targerts
targets = Series()
ages = read_ages('KJPP')
for session in features.index:
    age = ages[session]
    targets.loc[session] = age

# Drop targets with missing values
targets = targets.dropna()
features = features.loc[targets.index]

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
with open('old_model_80feat.pkl', 'rb') as file:
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
plt.scatter(accurate_x, accurate_y, color="#34AC8B", marker='.', alpha=0.3)
plt.scatter(inaccurate_x, inaccurate_y, color="#34AC8B", marker='.', alpha=0.3)
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






