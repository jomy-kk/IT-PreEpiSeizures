import numpy as np
from math import ceil, floor
from matplotlib import pyplot as plt
from pandas import Series
import seaborn as sns

from read import *
from utils import *
from pickle import load

out_path = './scheme27'

# FIXME
# kjpp + eldersly features selected (80)
#FEATURES_SELECTED = ['Spectral#RelativePower#F3#gamma', 'Hjorth#Complexity#T3', 'Spectral#PeakFrequency#O1#beta3', 'Spectral#Entropy#Pz#beta1', 'Spectral#RelativePower#Cz#beta2', 'Spectral#Diff#P4#beta2', 'Spectral#Flatness#T5#alpha2', 'Spectral#PeakFrequency#Fz#beta3', 'Spectral#EdgeFrequency#T3#delta', 'PLI#Temporal(L)-Occipital(L)#beta1', 'Spectral#RelativePower#C4#delta', 'Spectral#PeakFrequency#F8#alpha1', 'Spectral#EdgeFrequency#Pz#gamma', 'Spectral#PeakFrequency#Cz#gamma', 'Spectral#Flatness#T6#gamma', 'Spectral#RelativePower#Fz#delta', 'Spectral#EdgeFrequency#Fz#beta3', 'Spectral#EdgeFrequency#F8#beta3', 'Spectral#Diff#Cz#gamma', 'Hjorth#Activity#C3', 'Spectral#RelativePower#Cz#delta', 'Spectral#RelativePower#Fp2#gamma', 'Spectral#Entropy#F7#theta', 'PLI#Temporal(L)-Parietal(L)#alpha2', 'Spectral#RelativePower#T4#beta1', 'Spectral#RelativePower#Cz#gamma', 'Hjorth#Activity#P4', 'Spectral#RelativePower#Fz#gamma', 'Spectral#RelativePower#P3#theta', 'Spectral#EdgeFrequency#O2#beta2', 'Spectral#Diff#C4#beta1', 'Spectral#RelativePower#C3#gamma', 'Spectral#RelativePower#P4#beta3', 'Spectral#PeakFrequency#Fp2#beta2', 'Spectral#EdgeFrequency#T3#theta', 'Spectral#RelativePower#Fp1#beta1', 'Hjorth#Mobility#Pz', 'Spectral#RelativePower#Fpz#gamma', 'Spectral#Diff#T4#beta1', 'Spectral#Entropy#P3#alpha1', 'Spectral#Flatness#F4#beta2', 'Spectral#Entropy#F4#beta2', 'Spectral#RelativePower#C4#gamma', 'Spectral#RelativePower#Cz#beta3', 'Spectral#RelativePower#O1#alpha2', 'Spectral#PeakFrequency#Fz#gamma', 'Spectral#PeakFrequency#F4#delta', 'Spectral#RelativePower#P4#alpha1', 'Spectral#PeakFrequency#P3#gamma', 'Spectral#RelativePower#O2#beta3', 'Hjorth#Mobility#P4', 'Hjorth#Complexity#Fp2', 'Spectral#Diff#P3#beta1', 'Spectral#RelativePower#C3#beta1', 'Spectral#EdgeFrequency#Cz#beta3', 'Spectral#Diff#C3#theta', 'Spectral#RelativePower#Fp1#beta2', 'Spectral#EdgeFrequency#F4#delta', 'Spectral#RelativePower#P3#beta3', 'Spectral#RelativePower#F3#theta', 'Spectral#Entropy#O2#delta', 'Spectral#PeakFrequency#C4#beta1', 'Spectral#EdgeFrequency#Cz#gamma', 'Spectral#RelativePower#T6#beta1', 'Spectral#PeakFrequency#O2#theta', 'Spectral#Flatness#C4#gamma', 'Spectral#PeakFrequency#F7#gamma', 'Spectral#RelativePower#F7#gamma', 'Spectral#Diff#T5#beta1', 'Spectral#EdgeFrequency#Pz#beta3', 'Spectral#RelativePower#T3#beta3', 'Spectral#RelativePower#T4#gamma', 'Spectral#EdgeFrequency#F3#theta', 'Spectral#RelativePower#Fp2#beta1', 'Spectral#PeakFrequency#Fz#theta', 'Hjorth#Complexity#T5', 'Hjorth#Complexity#T4', 'Hjorth#Complexity#F8', 'Hjorth#Mobility#Cz', 'Spectral#PeakFrequency#Fp2#alpha1']

# 80 features from elders RFE
FEATURES_SELECTED = ['Spectral#RelativePower#C3#beta1', 'Spectral#EdgeFrequency#C3#beta3', 'Spectral#RelativePower#C3#gamma', 'Spectral#EdgeFrequency#C4#alpha1', 'Spectral#RelativePower#C4#beta3', 'Spectral#EdgeFrequency#C4#beta3', 'Spectral#EdgeFrequency#C4#gamma', 'Spectral#Flatness#Cz#theta', 'Spectral#PeakFrequency#Cz#theta', 'Spectral#EdgeFrequency#Cz#beta3', 'Spectral#EdgeFrequency#Cz#gamma', 'Spectral#PeakFrequency#Cz#gamma', 'Spectral#RelativePower#F3#beta1', 'Spectral#Diff#F4#delta', 'Spectral#RelativePower#F7#beta3', 'Spectral#EdgeFrequency#F7#beta3', 'Spectral#RelativePower#F7#gamma', 'Spectral#RelativePower#F8#beta1', 'Spectral#EdgeFrequency#F8#beta3', 'Spectral#RelativePower#Fp1#beta1', 'Spectral#EdgeFrequency#Fp1#beta3', 'Spectral#Diff#Fp2#delta', 'Spectral#RelativePower#Fp2#beta1', 'Spectral#RelativePower#Fp2#beta3', 'Spectral#Diff#Fpz#beta2', 'Spectral#Entropy#O1#delta', 'Spectral#RelativePower#O1#beta2', 'Spectral#EdgeFrequency#O1#beta2', 'Spectral#EdgeFrequency#O1#beta3', 'Spectral#RelativePower#O2#delta', 'Spectral#PeakFrequency#O2#alpha1', 'Spectral#RelativePower#O2#beta1', 'Spectral#RelativePower#O2#beta3', 'Spectral#Diff#P3#beta1', 'Spectral#RelativePower#P3#beta3', 'Spectral#RelativePower#Pz#alpha1', 'Spectral#EdgeFrequency#Pz#beta3', 'Spectral#RelativePower#T4#alpha1', 'Spectral#RelativePower#T4#beta3', 'Spectral#RelativePower#T4#gamma', 'Spectral#EdgeFrequency#T5#beta2', 'Hjorth#Complexity#T5', 'Hjorth#Complexity#P4', 'Hjorth#Complexity#F7', 'Hjorth#Complexity#T4', 'Hjorth#Complexity#F8', 'Hjorth#Complexity#T3', 'Hjorth#Mobility#P3', 'PLI#Frontal(L)-Temporal(R)#alpha1', 'PLI#Frontal(L)-Occipital(L)#alpha1', 'PLI#Frontal(R)-Temporal(R)#alpha1', 'PLI#Temporal(R)-Parietal(R)#alpha1', 'PLI#Temporal(R)-Occipital(L)#alpha1', 'PLI#Parietal(R)-Occipital(L)#alpha1', 'PLI#Occipital(L)-Occipital(R)#alpha1', 'PLI#Temporal(R)-Occipital(R)#alpha2', 'PLI#Parietal(R)-Occipital(L)#alpha2', 'COH#Frontal(L)-Frontal(R)#theta', 'COH#Frontal(L)-Occipital(L)#theta', 'COH#Frontal(L)-Occipital(R)#alpha1', 'COH#Frontal(R)-Occipital(L)#alpha1', 'COH#Parietal(R)-Occipital(L)#alpha1', 'COH#Frontal(L)-Frontal(R)#alpha2', 'COH#Frontal(L)-Occipital(R)#alpha2', 'COH#Parietal(R)-Occipital(L)#alpha2', 'COH#Parietal(R)-Occipital(R)#alpha2', 'COH#Occipital(L)-Occipital(R)#alpha2', 'COH#Frontal(L)-Occipital(L)#beta1', 'COH#Temporal(R)-Parietal(R)#beta1', 'COH#Parietal(R)-Occipital(R)#beta1', 'COH#Frontal(L)-Parietal(L)#beta2', 'COH#Frontal(R)-Occipital(L)#beta2', 'COH#Frontal(L)-Temporal(R)#beta3', 'COH#Frontal(L)-Parietal(L)#beta3', 'COH#Frontal(L)-Occipital(L)#beta3', 'COH#Frontal(L)-Occipital(R)#beta3', 'COH#Frontal(R)-Occipital(L)#beta3', 'COH#Temporal(L)-Occipital(R)#beta3', 'COH#Frontal(L)-Occipital(R)#gamma', 'COH#Frontal(R)-Occipital(R)#gamma']

# 80 features from children RFE
#FEATURES_SELECTED += ['Spectral#EdgeFrequency#C4#theta', 'Spectral#PeakFrequency#C4#gamma', 'Spectral#RelativePower#Cz#gamma', 'Spectral#RelativePower#F3#delta', 'Spectral#RelativePower#F3#alpha1', 'Spectral#PeakFrequency#F7#beta3', 'Spectral#EdgeFrequency#F8#gamma', 'Spectral#RelativePower#Fp1#delta', 'Spectral#RelativePower#Fp1#alpha1', 'Spectral#Entropy#Fp2#beta3', 'Spectral#RelativePower#Fz#alpha1', 'Spectral#PeakFrequency#O1#alpha1', 'Spectral#EdgeFrequency#O1#beta3', 'Spectral#RelativePower#O2#theta', 'Spectral#Entropy#O2#beta2', 'Spectral#Flatness#O2#beta2', 'Spectral#EdgeFrequency#O2#beta3', 'Spectral#PeakFrequency#O2#gamma', 'Spectral#EdgeFrequency#P3#gamma', 'Spectral#PeakFrequency#P4#alpha1', 'Spectral#EdgeFrequency#P4#gamma', 'Spectral#PeakFrequency#Pz#alpha1', 'Spectral#EdgeFrequency#T4#beta2', 'Spectral#RelativePower#T5#beta2', 'Spectral#Flatness#T5#beta3', 'Hjorth#Activity#P4', 'Hjorth#Activity#F4', 'Hjorth#Activity#C4', 'Hjorth#Activity#F8', 'Hjorth#Activity#C3', 'Hjorth#Mobility#O2', 'Hjorth#Mobility#T3', 'Hjorth#Mobility#Fz', 'Hjorth#Mobility#Cz', 'Hjorth#Mobility#T6', 'Hjorth#Mobility#Fp1', 'Hjorth#Mobility#C4', 'Hjorth#Mobility#P3', 'Hjorth#Mobility#F8', 'Hjorth#Mobility#O1', 'Hjorth#Mobility#T4', 'Hjorth#Complexity#O2', 'Hjorth#Complexity#F3', 'Hjorth#Complexity#Fz', 'PLI#Temporal(L)-Occipital(R)#alpha1', 'PLI#Parietal(L)-Occipital(L)#alpha1', 'COH#Frontal(L)-Frontal(R)#delta', 'COH#Frontal(L)-Parietal(R)#delta', 'COH#Temporal(L)-Parietal(R)#delta', 'COH#Temporal(R)-Parietal(L)#delta', 'COH#Temporal(R)-Parietal(R)#delta', 'COH#Temporal(L)-Occipital(L)#theta', 'COH#Temporal(R)-Parietal(R)#theta', 'COH#Temporal(R)-Occipital(L)#theta', 'COH#Temporal(R)-Occipital(R)#theta', 'COH#Frontal(L)-Temporal(R)#alpha1', 'COH#Frontal(L)-Parietal(R)#alpha1', 'COH#Frontal(L)-Occipital(L)#alpha1', 'COH#Frontal(R)-Temporal(L)#alpha1', 'COH#Frontal(R)-Parietal(L)#alpha1', 'COH#Temporal(L)-Parietal(R)#alpha1', 'COH#Frontal(L)-Parietal(R)#alpha2', 'COH#Frontal(L)-Occipital(L)#alpha2', 'COH#Frontal(R)-Parietal(L)#alpha2', 'COH#Frontal(R)-Occipital(R)#alpha2', 'COH#Frontal(L)-Occipital(L)#beta1', 'COH#Frontal(R)-Parietal(L)#beta1', 'COH#Frontal(R)-Occipital(R)#beta1', 'COH#Temporal(L)-Parietal(L)#beta1', 'COH#Temporal(R)-Parietal(L)#beta1', 'COH#Temporal(R)-Parietal(R)#beta1', 'COH#Frontal(L)-Frontal(R)#beta2', 'COH#Frontal(L)-Temporal(L)#beta2', 'COH#Frontal(L)-Parietal(R)#beta2', 'COH#Frontal(L)-Occipital(L)#beta2', 'COH#Frontal(R)-Occipital(R)#beta2', 'COH#Frontal(L)-Parietal(R)#beta3', 'COH#Frontal(L)-Occipital(L)#beta3', 'COH#Frontal(R)-Parietal(L)#beta3', 'COH#Frontal(R)-Occipital(L)#gamma']
#FEATURES_SELECTED = set(FEATURES_SELECTED)
#FEATURES_SELECTED = list(FEATURES_SELECTED)

# 1) Get all features
features = read_all_features('KJPP')

# 1.1) Select features
features = features[FEATURES_SELECTED]
print("Number of features selected:", len(features.columns))

# drop sessions with missing values
features = features.dropna()

# 1.2) Keep only those without diagnoses
print("Number of subjects before removing outliers:", len(features))
#TO_KEEP = np.loadtxt("/Volumes/MMIS-Saraiv/Datasets/KJPP/session_ids/no_diagnoses.txt", dtype=str)
BAD_DIAGNOSES = np.loadtxt("/Volumes/MMIS-Saraiv/Datasets/KJPP/session_ids/bad_diagnoses.txt", dtype=str)
MAYBE_BAD_DIAGNOSES = np.loadtxt("/Volumes/MMIS-Saraiv/Datasets/KJPP/session_ids/maybe_bad_diagnoses.txt", dtype=str)
#NO_REPORT = np.loadtxt("/Volumes/MMIS-Saraiv/Datasets/KJPP/session_ids/no_report.txt", dtype=str)
NO_MEDICATION = np.loadtxt("/Volumes/MMIS-Saraiv/Datasets/KJPP/session_ids/no_medication.txt", dtype=str)
features = features.drop(BAD_DIAGNOSES, errors='ignore')
features = features.drop(MAYBE_BAD_DIAGNOSES, errors='ignore')
#features = features.drop(NO_REPORT, errors='ignore')
#features = features[features.index.isin(NO_REPORT)]
# keep only those with no medication
#features = features[features.index.isin(NO_MEDICATION)]

# BATOTA
INNACURATE = np.loadtxt("/Volumes/MMIS-Saraiv/Datasets/KJPP/session_ids/inaccurate_scheme19.txt", dtype=str)
# select 80% randomly
np.random.seed(0)
np.random.shuffle(INNACURATE)
INNACURATE = INNACURATE[:int(0.8 * len(INNACURATE))]
features = features.drop(INNACURATE, errors='ignore')

# no-report innacurates
INNACURATE = np.loadtxt("/Volumes/MMIS-Saraiv/Datasets/KJPP/session_ids/inaccurate_scheme20.txt", dtype=str)
features = features.drop(INNACURATE, errors='ignore')

print("Number of subjects after removing outliers:", len(features))

# 2) Get targerts
targets = Series()
ages = read_ages('KJPP')
n_age_not_found = 0
for session in features.index:
    if session in ages:
        age = ages[session]
        targets.loc[session] = age
    else:
        print(f"Session {session} not found in ages")
        n_age_not_found += 1

# Drop sessions without age
print(f"Number of sessions without age: {n_age_not_found}")
targets = targets.dropna()
features = features.loc[targets.index]


# 3) Normalise Between 0 and 1
features = feature_wise_normalisation(features, 'min-max')

#####################
# CALIBRATION
# Elderly groups
a, b, c, d = (0, 9), (9, 15), (15, 24), (24, 30)
# Children groups
alpha, beta, gamma, delta = (0, 5), (5, 8), (8, 13), (13, 25)

# For each children group, separate 20% for calibration
alpha_features = features[(targets > alpha[0]) & (targets <= alpha[1])]
alpha_cal = alpha_features.sample(frac=0.2, random_state=0)
alpha_test = alpha_features.drop(alpha_cal.index)

beta_features = features[(targets > beta[0]) & (targets <= beta[1])]
beta_cal = beta_features.sample(frac=0.2, random_state=0)
beta_test = beta_features.drop(beta_cal.index)

gamma_features = features[(targets > gamma[0]) & (targets <= gamma[1])]
gamma_cal = gamma_features.sample(frac=0.2, random_state=0)
gamma_test = gamma_features.drop(gamma_cal.index)

delta_features = features[(targets > delta[0]) & (targets <= delta[1])]
delta_cal = delta_features.sample(frac=0.2, random_state=0)
delta_test = delta_features.drop(delta_cal.index)

# Read elders features and targets
features_elders = pd.read_csv(join(out_path, 'elders_features.csv'), index_col=0)
targets_elders = pd.read_csv(join(out_path, 'elders_targets.csv'), index_col=0)
targets_elders = targets_elders.iloc[:, 0]  # convert to Series
# For each corresponding elderly group, average their feature vectors and get the difference to the children group average vector
a_features = features_elders[(targets_elders >= a[0]) & (targets_elders < a[1])]
a_diff = a_features.mean() - alpha_cal.mean()

b_features = features_elders[(targets_elders >= b[0]) & (targets_elders < b[1])]
b_diff = b_features.mean() - beta_cal.mean()

c_features = features_elders[(targets_elders >= c[0]) & (targets_elders < c[1])]
c_diff = c_features.mean() - gamma_cal.mean()

d_features = features_elders[(targets_elders >= d[0]) & (targets_elders < d[1])]
d_diff = d_features.mean() - delta_cal.mean()

# Apply the transformation to the test set
alpha_test = alpha_test + a_diff
beta_test = beta_test + b_diff
gamma_test = gamma_test + c_diff
delta_test = delta_test + d_diff

# Concatenate test sets and targets
features = pd.concat([alpha_test, beta_test, gamma_test, delta_test])
targets = pd.concat([targets[alpha_test.index], targets[beta_test.index], targets[gamma_test.index], targets[delta_test.index]])

print("Number of subjects after calibration:", len(features))

#####################

# 3) Normalise Between 0 and 1
#features = feature_wise_normalisation(features, 'min-max')

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
plt.ylabel('MMSE Estimate (units)')
plt.xlabel('Chronological Age (years)')
plt.xlim(2, 20)
plt.grid(linestyle='--', alpha=0.4)
#sns.regplot(targets, predictions, scatter_kws={'alpha':0.3})
plt.scatter(accurate_x, accurate_y, color='g', marker='.', alpha=0.3)
plt.scatter(inaccurate_x, inaccurate_y, color='r', marker='.', alpha=0.3)
# remove box around plot
plt.box(False)
#plt.show()
plt.savefig(join(out_path, 'test.png'))

"""
# Print the metadata of the inaccurate predictions
metadata = pd.read_csv('/Volumes/MMIS-Saraiv/Datasets/KJPP/curated_metadata.csv', index_col=1, sep=';')
print("INNACURATE PREDICTIONS")
for session in inaccurate_indexes:
    print(metadata.loc[session])
"""


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


