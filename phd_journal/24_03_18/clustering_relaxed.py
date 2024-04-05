import numpy as np
from matplotlib import pyplot as plt
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split, cross_val_score, KFold

from read import *
from read import read_all_features
from utils import feature_wise_normalisation


############
# 1. Get all KJPP instances
kjpp = read_all_features('KJPP')

# 1.1. Drop subject_sessions with nans
kjpp = kjpp.dropna()

# 1.2. Select features
# FIXME
FEATURES_SELECTED = ['Spectral#Diff#C3#theta', 'Spectral#RelativePower#C3#gamma', 'Spectral#RelativePower#C4#delta', 'Spectral#PeakFrequency#C4#beta1', 'Spectral#Flatness#C4#gamma', 'Spectral#RelativePower#Cz#beta2', 'Spectral#RelativePower#Cz#beta3', 'Spectral#RelativePower#Cz#gamma', 'Spectral#Diff#Cz#gamma', 'Spectral#Entropy#F4#beta2', 'Spectral#Flatness#F4#beta2', 'Spectral#Entropy#F7#theta', 'Spectral#PeakFrequency#Fp2#alpha1', 'Spectral#PeakFrequency#Fp2#beta2', 'Spectral#RelativePower#Fz#delta', 'Spectral#PeakFrequency#Fz#theta', 'Spectral#PeakFrequency#Fz#gamma', 'Spectral#PeakFrequency#O1#beta3', 'Spectral#Entropy#O2#delta', 'Spectral#PeakFrequency#O2#theta', 'Spectral#PeakFrequency#P3#gamma', 'Spectral#Diff#P4#beta2', 'Spectral#EdgeFrequency#Pz#gamma', 'Spectral#EdgeFrequency#T3#delta', 'Spectral#Flatness#T5#alpha2', 'Hjorth#Activity#C3', 'Hjorth#Activity#P4', 'Hjorth#Mobility#Cz', 'PLI#Temporal(L)-Parietal(L)#alpha2', 'PLI#Temporal(L)-Occipital(L)#beta1']
kjpp = kjpp[FEATURES_SELECTED]
print("Number of features selected:", len(kjpp.columns))

# 1.3.  Remove outliers
# FIXME
print("Number of subjects before removing outliers:", len(kjpp))
OUTLIERS = [8,  40,  59, 212, 229, 247, 264, 294, 309, 356, 388, 391, 429, 437, 448, 460, 465, 512, 609, 653, 687, 688, 771, 808, 831, 872, 919]
kjpp = kjpp.drop(kjpp.index[OUTLIERS])
print("Number of subjects after removing outliers:", len(kjpp))

# 1.4. Normalise features
#kjpp = feature_wise_normalisation(kjpp, 'mean-std')
# Compute KJPP stochastic signature
# Mean and std for each feature
kjpp_min = kjpp.min()
kjpp_max = kjpp.max()
kjpp_mean = kjpp.mean()
kjpp_std = kjpp.std()

# 1.5. Get targets
kjpp_targets = read_ages('KJPP')
kjpp['targets'] = None
for index in kjpp.index:
    kjpp.loc[index, 'targets'] = kjpp_targets[index]
# 1.6. Drop subjects without targets
kjpp = kjpp.dropna()


############
# 2. Get all elderly instances
insight = read_all_features('INSIGHT')
insight = insight[FEATURES_SELECTED]
# create a new column "Dataset" and assign 'INSIGHT' to all rows
insight['Dataset'] = 'INSIGHT'
brainlat = read_all_features('BrainLat')
brainlat = brainlat[FEATURES_SELECTED]
brainlat['Dataset'] = 'BrainLat'
miltiadous = read_all_features('Miltiadous Dataset')
miltiadous = miltiadous[FEATURES_SELECTED]
miltiadous['Dataset'] = 'Miltiadous Dataset'
elderly = pd.concat([insight, brainlat, miltiadous], axis=0)

# 2.1. Drop subject_sessions with nans
elderly = elderly.dropna()

# 2.4. Get targets
insight_targets = read_mmse('INSIGHT')
brainlat_targets = read_mmse('BrainLat')
miltiadous_targets = read_mmse('Miltiadous Dataset')

elderly['targets'] = None
for index in elderly.index:
    if '_' in str(index):  # insight
        key = int(index.split('_')[0])
        if key in insight_targets:
            elderly.loc[index, 'targets'] = insight_targets[key]
    elif '-' in str(index):  # brainlat
        if index in brainlat_targets:
            elderly.loc[index, 'targets'] = brainlat_targets[index]
    else:  # miltiadous
        # parse e.g. 24 -> 'sub-024'; 1 -> 'sub-001'
        key = 'sub-' + str(index).zfill(3)
        if key:
            elderly.loc[index, 'targets'] = miltiadous_targets[key]

# 2.5. Drop subjects without targets
elderly = elderly.dropna()

def calibration(dataset, reference, A=None, B=None, C=None):
    targets = dataset['targets']
    dataset = dataset[FEATURES_SELECTED]

    # A. Pre-Normalisation; A1: auto-normalisation [0, 1]; A2: reference-normalisation [min_ref, max_ref]
    if A == 1:
        dataset = (dataset - dataset.min()) / (dataset.max() - dataset.min())
    elif A == 2:
        dataset = (dataset - dataset.min()) / (dataset.max() - dataset.min())
        dataset = dataset * (reference.max() - reference.min()) + reference.min()

    # B. Ages Stochastic Signature
    # None: no calibration
    # [((mmse_ref_min, mmse_ref_max, age_ref_min, age_ref_max), (mmse_apply_min, mmse_apply_max)), ...]: calibration by reference

    if B is not None:
        for ref_points, apply_interval in B:
            mmse_ref_min, mmse_ref_max, age_ref_min, age_ref_max = ref_points
            mmse_ref_dataset = dataset[(mmse_ref_min <= dataset['targets']) & (dataset['targets'] <= mmse_ref_max)]
            mmse_ref_dataset = mmse_ref_dataset[FEATURES_SELECTED]

            kjpp_ref_dataset = reference[(age_ref_min <= reference['targets']) & (reference['targets'] <= age_ref_max)]
            kjpp_ref_dataset_mean = kjpp_ref_dataset.mean()
            kjpp_ref_dataset_std = kjpp_ref_dataset.std()

            for feature in mmse_ref_dataset.columns:
                old_mean = mmse_ref_dataset[feature].mean()
                old_std = mmse_ref_dataset[feature].std()
                new_mean = kjpp_ref_dataset_mean[feature]
                new_std = kjpp_ref_dataset_std[feature]
                # transform
                mmse_ref_dataset[feature] = (mmse_ref_dataset[feature] - old_mean) * (new_std / old_std) + new_mean

            # Understand the transformation done to reference MMSe point and apply it to the remaining of the dataset
            before = dataset[(dataset['targets'] <= mmse_ref_max) & (dataset['targets'] >= mmse_ref_min)]
            before = before[FEATURES_SELECTED]
            # Get the difference
            diff = mmse_ref_dataset.mean() - before.mean()

            # Apply the difference to the rest of the dataset
            mmse_apply_min, mmse_apply_max = apply_interval
            mmse_non_ref = dataset[(dataset['targets'] <= mmse_apply_max) & (dataset['targets'] >= mmse_apply_min)]
            mmse_non_ref = mmse_non_ref[FEATURES_SELECTED]
            mmse_non_ref = mmse_non_ref + diff

            # unaffected part of the dataset
            indexes_affected = mmse_non_ref.index.union(mmse_ref_dataset.index)
            mmse_unaffected = dataset.drop(indexes_affected)

            # Concatenate
            dataset = pd.concat([mmse_unaffected, mmse_ref_dataset, mmse_non_ref])
            # add the targets back
            dataset['targets'] = targets

    # C. Normalisation; B1: auto-normalisation [0, 1]; B2: reference-normalisation [min_ref, max_ref]
    if C == 1:
        dataset = (dataset - dataset.min()) / (dataset.max() - dataset.min())
    elif C == 2:
        dataset = (dataset - dataset.min()) / (dataset.max() - dataset.min())
        dataset = dataset * (reference.max() - reference.min()) + reference.min()

    return dataset


# 3. Make some calibrations
A, C = 2, 2

# 3.1. Adjust INSIGHT subjects with perfect MMSE to KJPP adults
B = [((30, 30, 17.5, 25), (0, 29))]
insight = elderly[elderly['Dataset'] == 'INSIGHT']
insight = calibration(insight, kjpp, A, B, C)

# 3.2. Adjust BrainLat subjects with 24<=MMSE<=30 to KJPP children aged 13-19
B = [((24, 30, 13, 19), (0, 23))]
brainlat = elderly[elderly['Dataset'] == 'BrainLat']
brainlat = calibration(brainlat, kjpp, A, B, C)

# 3.3. Adjust Miltiadous subjects with perfect MMSE to KJPP adults
B = [((30, 30, 17.5, 25), (0, 29))]
miltiadous = elderly[elderly['Dataset'] == 'Miltiadous Dataset']
miltiadous = calibration(miltiadous, kjpp, A, B, C)

# 3.4. Concatenate all datasets
elderly = pd.concat([insight, brainlat, miltiadous], axis=0)

# Remove targets and Dataset from elderly DataFrame
elderly_targets = elderly['targets']
elderly = elderly.drop(columns=['targets'])

# Remove targets from KJPP DataFrame
kjpp_targets = kjpp['targets']
kjpp = kjpp.drop(columns='targets')


"""
In this project, we are interested in assigning the the elderly instances to the KJPP clusters they are more closed to.
For that, we'll use a measure of distance between the feature vectors, and assign each elderly instance to the cluster
where it has the 5 nearest neighbours.

The clusters are pre-defined (no model):
- Cluster 0: MMSE 0-24; Developmental Age 0-13
- Cluster 2: MMSE 24-30; Developmental Age 13-24
"""


def get_cluster_by_age(age: float) -> int:
    if age <= 13:
        return 0
    else:
        return 1


def get_cluster_by_mmse(mmse: int) -> int:
    if mmse <= 24:
        return 0
    else:
        return 1


def distance(x: np.ndarray, y: np.ndarray, measure:str = 'euclidean') -> float:
    if measure == 'euclidean':
        return np.linalg.norm(x - y)
    elif measure == 'cosine':
        return np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y))
    elif measure == 'manhattan':
        return np.sum(np.abs(x - y))
    elif measure == 'chebyshev':
        return np.max(np.abs(x - y))
    elif measure == 'minkowski':
        p = 1.5
        return np.sum(np.abs(x - y) ** p) ** (1/p)
    else:
        raise ValueError("Invalid measure")


def assign_cluster_based_on_nn(elderly_features: np.array, k=5, measure='euclidean'):
    """Given a feature vector of an elderly instance, assign it to the cluster where it has the k nearest KJPP neighbours."""
    distances = []
    for kjpp_features in kjpp.values:
        distances.append(distance(elderly_features, kjpp_features, measure))
    nearest_neighbours = np.argsort(distances)[:k]
    # Get the mean age of the nearest neighbours
    mean_age = np.median([kjpp_targets[kjpp.index[i]] for i in nearest_neighbours])
    return get_cluster_by_age(mean_age)


# 3. Assign clusters
K = 5
MEASURE = 'minkowski'
print(f"K={K}, measure={MEASURE}")
predicted_elderly_clusters = []
for elderly_features in elderly.values:
    predicted_elderly_clusters.append(assign_cluster_based_on_nn(elderly_features, k=K, measure=MEASURE))
elderly['cluster'] = predicted_elderly_clusters

# 3. Expected ground-truth clusters
true_elderly_clusters = [get_cluster_by_mmse(mmse) for mmse in elderly_targets]

# 4. Plot confusion matrix in percentage of size MMSE clusters
confusion_matrix = np.zeros((2, 2))
for true, pred in zip(true_elderly_clusters, predicted_elderly_clusters):
    confusion_matrix[true, pred] += 1
#confusion_matrix = confusion_matrix / np.sum(confusion_matrix, axis=1)[:, None]  # uncomment for relative values
plt.imshow(confusion_matrix, cmap='Blues')
# Write percentages on the plot
for i in range(2):
    for j in range(2):
        plt.text(j, i, f"{confusion_matrix[i, j]:.2f}", ha='center', va='center', color='black')
plt.colorbar()
plt.xlabel('Found Similarity to Age Group...')
plt.xticks([0, 1], ['0-13', '13-24'])
plt.ylabel('MMSE')
plt.yticks([0, 1], ['0-24', '24-30'])
plt.title(f"K={K}, measure={MEASURE}")
plt.show()

# 5. Evaluate
print("Accuracy:", np.trace(confusion_matrix) / np.sum(confusion_matrix))
print("Precision:", np.diag(confusion_matrix) / np.sum(confusion_matrix, axis=0))
print("Recall:", np.diag(confusion_matrix) / np.sum(confusion_matrix, axis=1))
print("F1-Score:", 2 * np.diag(confusion_matrix) / (np.sum(confusion_matrix, axis=0) + np.sum(confusion_matrix, axis=1)))
print("F1-Score weighted by cluster sizes:", np.sum(np.diag(confusion_matrix) * np.sum(confusion_matrix, axis=1)) / np.sum(confusion_matrix))

# Compute chi-squared test between true and predicted clusters
# H0: The true and predicted clusters are independent
# H1: The true and predicted clusters are dependent
# If p-value < 0.05, we reject H0

# Note: matrix cannot contain zeros. let's add 1 to all cells
confusion_matrix += 1

from scipy.stats import chi2_contingency
chi2, p, dof, ex = chi2_contingency(confusion_matrix)
print("Chi-squared test p-value:", p)
print("Chi-squared test statistic:", chi2)
print("Degrees of freedom:", dof)
print("Expected frequencies:", ex)



