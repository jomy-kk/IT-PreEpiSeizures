import numpy as np
from matplotlib import pyplot as plt
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split, cross_val_score, KFold

from read import *
from read import read_all_features
from utils import feature_wise_normalisation


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

# 1.5. Get targets
kjpp_targets = read_ages('KJPP')



# 2. Get all elderly instances
insight = read_all_features('INSIGHT')
insight = insight[FEATURES_SELECTED]
brainlat = read_all_features('BrainLat')
brainlat = brainlat[FEATURES_SELECTED]
miltiadous = read_all_features('Miltiadous Dataset')
miltiadous = miltiadous[FEATURES_SELECTED]
elderly = pd.concat([insight, brainlat, miltiadous], axis=0)

# 2.1. Drop subject_sessions with nans
elderly = elderly.dropna()

# 2.3. Normalise each feature column to have the same max and min as the KJPP
# So, the min of elderly needs to be the min of KJPP, and the max of elderly needs to be the max of KJPP
elderly = (elderly - elderly.min()) / (elderly.max() - elderly.min())
elderly = elderly * (kjpp_max - kjpp_min) + kjpp_min


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
elderly_targets = elderly['targets']
elderly = elderly.drop(columns='targets')


"""
In this project, we are interested in assigning the the elderly instances to the KJPP clusters they are more closed to.
For that, we'll use a measure of distance between the feature vectors, and assign each elderly instance to the cluster
where it has the 5 nearest neighbours.

The clusters are pre-defined (no model):
- Cluster 0: MMSE 0-9; Developmental Age 0-5
- Cluster 1: MMSE 9-15; Developmental Age 5-8
- Cluster 2: MMSE 15-24; Developmental Age 8-13
- Cluster 3: MMSE 24-30; Developmental Age 13-24
"""


def get_cluster_by_age(age: float) -> int:
    if age <= 5:
        return 0
    elif age <= 8:
        return 1
    elif age <= 13:
        return 2
    else:
        return 3


def get_cluster_by_mmse(mmse: int) -> int:
    if mmse <= 9:
        return 0
    elif mmse <= 15:
        return 1
    elif mmse <= 24:
        return 2
    else:
        return 3


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
        return np.sum((x - y) ** 2) ** (1/2)
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
confusion_matrix = np.zeros((4, 4))
for true, pred in zip(true_elderly_clusters, predicted_elderly_clusters):
    confusion_matrix[true, pred] += 1
#confusion_matrix = confusion_matrix / np.sum(confusion_matrix, axis=1)[:, None]
plt.imshow(confusion_matrix, cmap='Blues')
# Write percentages on the plot
for i in range(4):
    for j in range(4):
        plt.text(j, i, f"{confusion_matrix[i, j]}", ha='center', va='center', color='black')
plt.colorbar()
plt.xlabel('Found Similarity to Age Group...')
plt.xticks([0, 1, 2, 3], ['0-5', '5-8', '8-13', '13-24'])
plt.ylabel('MMSE')
plt.yticks([0, 1, 2, 3], ['0-9', '9-15', '15-24', '24-30'])
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



