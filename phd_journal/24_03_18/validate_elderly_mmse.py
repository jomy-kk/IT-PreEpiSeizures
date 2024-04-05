import numpy as np
from math import ceil, floor
from matplotlib import pyplot as plt
from pandas import Series
import seaborn as sns

from read import *
from utils import *

# FIXME
FEATURES_SELECTED = ['Spectral#Diff#C3#theta', 'Spectral#RelativePower#C3#gamma', 'Spectral#RelativePower#C4#delta', 'Spectral#PeakFrequency#C4#beta1', 'Spectral#Flatness#C4#gamma', 'Spectral#RelativePower#Cz#beta2', 'Spectral#RelativePower#Cz#beta3', 'Spectral#RelativePower#Cz#gamma', 'Spectral#Diff#Cz#gamma', 'Spectral#Entropy#F4#beta2', 'Spectral#Flatness#F4#beta2', 'Spectral#Entropy#F7#theta', 'Spectral#PeakFrequency#Fp2#alpha1', 'Spectral#PeakFrequency#Fp2#beta2', 'Spectral#RelativePower#Fz#delta', 'Spectral#PeakFrequency#Fz#theta', 'Spectral#PeakFrequency#Fz#gamma', 'Spectral#PeakFrequency#O1#beta3', 'Spectral#Entropy#O2#delta', 'Spectral#PeakFrequency#O2#theta', 'Spectral#PeakFrequency#P3#gamma', 'Spectral#Diff#P4#beta2', 'Spectral#EdgeFrequency#Pz#gamma', 'Spectral#EdgeFrequency#T3#delta', 'Spectral#Flatness#T5#alpha2', 'Hjorth#Activity#C3', 'Hjorth#Activity#P4', 'Hjorth#Mobility#Cz', 'PLI#Temporal(L)-Parietal(L)#alpha2', 'PLI#Temporal(L)-Occipital(L)#beta1']

# 1) Get all features
insight = read_all_features('INSIGHT')
insight = insight[FEATURES_SELECTED]
brainlat = read_all_features('BrainLat')
brainlat = brainlat[FEATURES_SELECTED]
miltiadous = read_all_features('Miltiadous Dataset')
miltiadous = miltiadous[FEATURES_SELECTED]
elderly = pd.concat([insight, brainlat, miltiadous], axis=0)

# Drop subject_sessions with nans
elderly = elderly.dropna()


# 2) Normalise features

# 2.1) With mean-std of KJPP
#elderly = feature_wise_normalisation_with_coeffs(elderly, 'mean-std', 'kjpp_stochastic_pattern.csv')
elderly = feature_wise_normalisation(elderly, 'mean-std')


# 3) Get all targets
insight_targets = read_mmse('INSIGHT')
brainlat_targets = read_mmse('BrainLat')
miltiadous_targets = read_mmse('Miltiadous Dataset')
targets = Series()
for index in elderly.index:
    if '_' in str(index):  # insight
        key = int(index.split('_')[0])
        if key in insight_targets:
            targets.loc[index] = insight_targets[key]
    elif '-' in str(index):  # brainlat
        if index in brainlat_targets:
            targets.loc[index] = brainlat_targets[index]
    else:  # miltiadous
        # parse e.g. 24 -> 'sub-024'; 1 -> 'sub-001'
        key = 'sub-' + str(index).zfill(3)
        if key:
            targets.loc[index] = miltiadous_targets[key]

# Get rid of nans
targets = targets.dropna()
elderly = elderly.loc[targets.index]


# 2.2) Calibrate features with target == 30 making them have the same mean and std of KJPP adults
cal_ref = elderly[targets == 30]
adult_stochastics = read_csv('kjpp_adult_stochastic_pattern.csv', index_col=0)

for feature in cal_ref.columns:
    old_mean = cal_ref[feature].mean()
    old_std = cal_ref[feature].std()
    new_mean = adult_stochastics[feature]['mean']
    new_std = adult_stochastics[feature]['std']
    # transform
    cal_ref[feature] = (cal_ref[feature] - old_mean) * (new_std / old_std) + new_mean

# Understand the transformation done to reference and apply it to the remaining of the dataset
before = elderly[targets == 30]
diff_mean = cal_ref.mean() - before.mean()

# Apply the difference to the rest of the dataset
cal_non_ref = elderly[targets < 30]
cal_non_ref = cal_non_ref + diff_mean

# Concatenate
elderly = pd.concat([cal_ref, cal_non_ref])

"""
# 2.3) With mean-std of KJPP AGAIN
elderly = feature_wise_normalisation(elderly, 'mean-std')
"""

# 3) Data Augmentation in the underrepresented MMSE groups
# MMSE groups: 0-9, 9-15, 15-20, 20-24, 24-26, 26-30
# We'll augment the 0-9, 9-15, 15-20, 20-24, 24-26 groups because they are underrepresented
# We'll augment them until they have the same number of samples as the 26-30 group

# Get the number of samples in each group
mmse_groups = ((0, 9), (10, 15), (16, 20), (21, 25), (26, 30))
mmse_groups_samples = [len(targets[(mmse[0] <= targets) & (targets <= mmse[1])]) for mmse in mmse_groups]
max_samples = max(mmse_groups_samples)

# Augment the underrepresented groups
for i, mmse_group in enumerate(mmse_groups):
    if mmse_groups_samples[i] < max_samples:
        # Get the number of samples to augment
        n_samples_to_augment = max_samples - mmse_groups_samples[i]
        # Get the samples to augment
        samples = elderly[targets.between(mmse_group[0], mmse_group[1])]
        # Augment with gaussian noise with sensitivity S
        S = 0.1
        i = 0
        n_cycles = 1
        while n_samples_to_augment > 0:
            # Augment
            augmented = samples.iloc[i] + np.random.normal(0, S, len(samples.columns))
            name = str(samples.index[i]) + '_augmented_' + str(n_cycles)
            # Append
            elderly.loc[name] = augmented
            targets.loc[name] = targets[samples.index[i]]
            # Update
            i += 1
            n_samples_to_augment -= 1
            if i == len(samples):
                i = 0
                n_cycles += 1

# Assert that the number of samples in each group is the same
print("Number of samples before augmentation:")
print(mmse_groups)
print(mmse_groups_samples)
mmse_groups_samples_after = [len(targets[(mmse[0] <= targets) & (targets <= mmse[1])]) for mmse in mmse_groups]
assert all([samples == max_samples for samples in mmse_groups_samples_after])
print("Number of samples after augmentation:")
print(mmse_groups)
print(mmse_groups_samples_after)

# 4) Load model
from pickle import load
with open('model.pkl', 'rb') as file:
    model = load(file)

# 5) Predict
predictions = model.predict(elderly)

# 6) Plot

def is_good_developmental_age_estimate(estimate: float, mmse: int) -> bool:
    """
    Outputs a MMSE approximation given the developmental age estimated by an EEG model.
    """
    assert 0 <= mmse <= 30, "MMSE must be between 0 and 30"
    assert 0 <= estimate, "Developmental age estimate must be positive"

    if estimate < 1.25:
        return 0 <= mmse <= estimate / 2
    elif estimate < 2:
        return floor((4 * estimate / 15) - (1 / 3)) <= mmse <= ceil(estimate / 2)
    elif estimate < 5:
        return (4 * estimate / 15) - (1 / 3) <= mmse <= 2 * estimate + 5
    elif estimate < 7:
        return 2 * estimate - 6 <= mmse <= (4 * estimate / 3) + (25 / 3)
    elif estimate < 8:
        return (4 * estimate / 5) + (47 / 5) <= mmse <= (4 * estimate / 3) + (25 / 3)
    elif estimate < 12:
        return (4 * estimate / 5) + (47 / 5) <= mmse <= (4 * estimate / 5) + (68 / 5)
    elif estimate < 13:
        return (4 * estimate / 7) + (92 / 7) <= mmse <= (4 * estimate / 5) + (68 / 5)
    elif estimate < 19:
        return (4 * estimate / 7) + (92 / 7) <= mmse <= 30
    elif estimate >= 19:
        return mmse >= 29


accurate = []
inaccurate = []
for prediction, mmse in zip(predictions, targets):
    if is_good_developmental_age_estimate(prediction, mmse):
        accurate.append((prediction, mmse))
    else:
        inaccurate.append((prediction, mmse))

accurate_x, accurate_y = zip(*accurate)
inaccurate_x, inaccurate_y = zip(*inaccurate)

# 9. Plot predictions vs targets
plt.figure()
plt.xlabel('Developmental Age Estimate (years)')
plt.ylabel('Acceptable MMSE (unit)')
plt.xticks((0, 1, 2, 5, 7, 8, 12, 13, 19, 25))
plt.yticks((0, 1, 9, 15, 19, 20, 24, 26, 27, 29, 30))
plt.xlim(0, 25.1)
plt.ylim(-0.5, 30.5)
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

# pearson rank correlation
from scipy.stats import pearsonr
pearson, pvalue = pearsonr(targets, predictions)
print("Pearson rank correlation:", pearson, f"(p={pvalue})")

# 10.1. Rank correlation
from scipy.stats import spearmanr
spearman, pvalue = spearmanr(targets, predictions, alternative='greater')
print("Spearman rank correlation:", spearman, f"(p={pvalue})")

# Other rank correlations
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
prediction_classes = ((0, 5), (5, 8), (8, 13), (13, 25))
mmse_classes = ((0, 9), (9, 15), (15, 24), (24, 30))

# assign predictions to classes
prediction_classes_assigned = []
for prediction in predictions:
    for i, (lower, upper) in enumerate(prediction_classes):
        if lower <= float(prediction) <= upper:
            prediction_classes_assigned.append(i)
            break
# assign targets to classes
mmse_classes_assigned = []
for mmse in targets:
    for i, (lower, upper) in enumerate(mmse_classes):
        if lower <= mmse <= upper:
            mmse_classes_assigned.append(i)
            break

# confusion matrix
conf_matrix = confusion_matrix(mmse_classes_assigned, prediction_classes_assigned)
# plot
plt.figure()
sns.heatmap(conf_matrix, annot=True, cmap='Blues', fmt='g',
            xticklabels=[f'{lower}-{upper}' for lower, upper in mmse_classes],
            yticklabels=[f'{lower}-{upper}' for lower, upper in prediction_classes])
plt.xlabel('Developmental Age Estimate (years)')
plt.ylabel('MMSE (units)')
plt.show()






