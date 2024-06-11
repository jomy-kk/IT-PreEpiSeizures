import numpy as np
from math import ceil, floor
from matplotlib import pyplot as plt
from pandas import Series
import seaborn as sns

from read import *
from utils import *
from pickle import load

out_path = './scheme1'

# 1) Get all features
miltiadous = read_all_features('Miltiadous Dataset', multiples=True)
brainlat = read_all_features('BrainLat', multiples=True)
sapienza = read_all_features('Sapienza', multiples=True)
insight = read_all_features('INSIGHT', multiples=True)
features = pd.concat([brainlat, miltiadous, sapienza, insight], axis=0)
print("Read all features. Shape:", features.shape)

# new: MI + RFE (932 -> 200 -> 80 features)
FEATURES_SELECTED = ['Spectral#PeakFrequency#C3#delta', 'Spectral#Entropy#C3#theta', 'Spectral#Flatness#C3#alpha', 'Spectral#EdgeFrequency#C3#alpha', 'Spectral#PeakFrequency#C3#alpha', 'Spectral#Diff#C3#alpha', 'Spectral#Entropy#C3#beta', 'Spectral#Diff#C3#beta', 'Spectral#Entropy#C3#gamma', 'Spectral#Flatness#C3#gamma', 'Spectral#EdgeFrequency#C4#delta', 'Spectral#PeakFrequency#C4#delta', 'Spectral#Diff#C4#delta', 'Spectral#RelativePower#C4#theta', 'Spectral#Flatness#C4#theta', 'Spectral#Flatness#C4#alpha', 'Spectral#EdgeFrequency#C4#alpha', 'Spectral#PeakFrequency#C4#alpha', 'Spectral#RelativePower#C4#beta', 'Spectral#Entropy#C4#beta', 'Spectral#RelativePower#C4#gamma', 'Spectral#Flatness#C4#gamma', 'Spectral#PeakFrequency#C4#gamma', 'Spectral#Diff#C4#gamma', 'Spectral#RelativePower#Cz#delta', 'Spectral#Flatness#Cz#delta', 'Spectral#Diff#Cz#delta', 'Spectral#RelativePower#Cz#theta', 'Spectral#Flatness#Cz#theta', 'Spectral#PeakFrequency#Cz#theta', 'Spectral#Entropy#Cz#alpha', 'Spectral#EdgeFrequency#Cz#alpha', 'Spectral#RelativePower#Cz#beta', 'Spectral#EdgeFrequency#Cz#beta', 'Spectral#Flatness#Cz#gamma', 'Spectral#EdgeFrequency#Cz#gamma', 'Spectral#RelativePower#F3#delta', 'Spectral#Flatness#F3#delta', 'Spectral#RelativePower#F3#theta', 'Spectral#Entropy#F3#theta', 'Spectral#EdgeFrequency#F3#theta', 'Spectral#PeakFrequency#F3#theta', 'Spectral#Diff#F3#theta', 'Spectral#RelativePower#F3#beta', 'Spectral#EdgeFrequency#F3#beta', 'Spectral#PeakFrequency#F3#beta', 'Spectral#EdgeFrequency#F3#gamma', 'Spectral#Diff#F3#gamma', 'Spectral#RelativePower#F4#delta', 'Spectral#Diff#F4#theta', 'Spectral#PeakFrequency#F4#alpha', 'Spectral#RelativePower#F4#gamma', 'Spectral#Entropy#F4#gamma', 'Spectral#PeakFrequency#F4#gamma', 'Spectral#RelativePower#F7#delta', 'Spectral#Entropy#F7#delta', 'Spectral#Flatness#F7#delta', 'Spectral#EdgeFrequency#F7#delta', 'Spectral#Entropy#F7#theta', 'Spectral#EdgeFrequency#F7#theta', 'Spectral#Diff#F7#theta', 'Spectral#Entropy#F7#alpha', 'Spectral#Entropy#F7#beta', 'Spectral#EdgeFrequency#F7#beta', 'Spectral#PeakFrequency#F7#beta', 'Spectral#Entropy#F7#gamma', 'Spectral#EdgeFrequency#F7#gamma', 'Spectral#RelativePower#F8#delta', 'Spectral#Flatness#F8#delta', 'Spectral#EdgeFrequency#F8#delta', 'Spectral#Entropy#F8#theta', 'Spectral#EdgeFrequency#F8#theta', 'Spectral#PeakFrequency#F8#theta', 'Spectral#Diff#F8#theta', 'Spectral#RelativePower#F8#alpha', 'Spectral#Flatness#F8#alpha', 'Spectral#EdgeFrequency#F8#alpha', 'Spectral#PeakFrequency#F8#alpha', 'Spectral#RelativePower#F8#beta', 'Spectral#Entropy#F8#beta']
features = features[FEATURES_SELECTED]
features = features.dropna(axis=0)

# 2) Get targets
insight_targets = read_mmse('INSIGHT')
brainlat_targets = read_mmse('BrainLat')
miltiadous_targets = read_mmse('Miltiadous Dataset')
sapienza_targets = read_mmse('Sapienza')
targets = Series()
batch = []
for index in features.index:
    if '$' in str(index):  # Multiples
        key = str(index).split('$')[0]  # remove the multiple
    else:  # Original
        key = index

    if '_' in str(key):  # insight
        key = int(key.split('_')[0])
        if key in insight_targets:
            targets.loc[index] = insight_targets[key]
            batch.append(1)
    elif '-' in str(key):  # brainlat
        if key in brainlat_targets:
            targets.loc[index] = brainlat_targets[key]
            batch.append(2)
    elif 'PARTICIPANT' in str(key):  # sapienza
        if key in sapienza_targets:
            targets.loc[index] = sapienza_targets[key]
            batch.append(3)
    else:  # miltiadous
        # parse e.g. 24 -> 'sub-024'; 1 -> 'sub-001'
        key = 'sub-' + str(key).zfill(3)
        if key:
            targets.loc[index] = miltiadous_targets[key]
            batch.append(4)

# Drop subject_sessions with nans targets
targets = targets.dropna()
features = features.loc[targets.index]

# Normalise feature vectors BEFORE
features = feature_wise_normalisation(features, method='min-max')
features = features.dropna(axis=1)


# 3) Convert features to an appropriate format
feature_names = features.columns.to_numpy()
sessions = features.index.to_numpy()
features = [features.loc[code].to_numpy() for code in sessions]

#Size
print("Number of subjects:", len(features))
print("Number of features:", len(features[0]))


# 4) Load model
from pickle import load
with open(join(out_path, 'model.pkl'), 'rb') as file:
    model = load(file)

# 5) Predict
predictions = model.predict(features)


def is_good_developmental_age_estimate(age: float, mmse: int, margin:float=0) -> bool:
    """
    Checks if the MMSE estimate is within the acceptable range for the given age.
    A margin can be added to the acceptable range.
    """
    #assert 0 <= mmse <= 30, "MMSE must be between 0 and 30"
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
inaccurate_indexes = []
for i, (prediction, age) in enumerate(zip(predictions, targets)):
    if is_good_developmental_age_estimate(age, prediction, margin=1.5):
        accurate.append((age, prediction))
    else:
        inaccurate.append((age, prediction))
        inaccurate_indexes.append(sessions[i])

accurate_x, accurate_y = zip(*accurate)
inaccurate_x, inaccurate_y = zip(*inaccurate)

# 9. Plot predictions vs targets
plt.figure()
plt.ylabel('Age Estimate (years)')
plt.xlabel('MMSE (units)')
plt.xlim(0, 30)
plt.grid(linestyle='--', alpha=0.4)
#sns.regplot(targets, predictions, scatter_kws={'alpha':0.3})
plt.scatter(accurate_x, accurate_y, color='g', marker='.', alpha=0.3)
plt.scatter(inaccurate_x, inaccurate_y, color='r', marker='.', alpha=0.3)
# remove box around plot
plt.box(False)
#plt.show()
plt.savefig(join(out_path, 'test.png'))

# Print the metadata of the inaccurate predictions
#metadata = pd.read_csv('/Volumes/MMIS-Saraiv/Datasets/KJPP/curated_metadata.csv', index_col=1, sep=';')
#print("INNACURATE PREDICTIONS")
#for session in inaccurate_indexes:
#    print(metadata.loc[session])
# indexes to numpy
#inaccurate_indexes = np.array(inaccurate_indexes)
#np.savetxt(join(out_path, 'noreport_inaccurate.txt'), inaccurate_indexes, fmt='%s')



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
"""
from scipy.stats import somersd
res = somersd(targets, predictions)
correlation, pvalue, table = res.statistic, res.pvalue, res.table
print("Somers' D:", correlation, f"(p={pvalue})")
"""

# Confusion Matrix
"""
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

"""
