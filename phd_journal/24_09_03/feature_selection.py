import numpy as np
from imblearn.over_sampling import SMOTE
from matplotlib import pyplot as plt
from pandas import Series
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.feature_selection import RFE, VarianceThreshold, SelectKBest, SelectPercentile, r_regression, f_regression, \
    mutual_info_regression, RFECV
from sklearn.model_selection import train_test_split, cross_val_score, KFold, ShuffleSplit

from read import *
from read import read_all_features
from utils import feature_wise_normalisation


def rfe_selection(model, features, targets, n_features_to_select, step, feature_names):
    """
    Recursive feature elimination
    """
    selector = RFE(estimator=model, n_features_to_select=n_features_to_select, step=step, verbose=2)
    features_transformed = selector.fit_transform(features, targets)

    # Get indices of the selected features
    scores = selector.ranking_
    indices = tuple(selector.get_support(indices=True))
    sorted_indices = sorted(indices, key=lambda i: scores[i], reverse=True)

    print("Selected features (in descending order of score):")
    print(sorted_indices)
    if feature_names is not None:
        print([feature_names[i] for i in sorted_indices])

    return features_transformed, indices


def rfe_selection_cv(model, features, targets, step, feature_names):
    """
    Recursive feature elimination
    """
    selector = RFECV(estimator=model, step=step, n_jobs=-1, verbose=2)
    features_transformed = selector.fit_transform(features, targets)

    # Get indices of the selected features
    scores = selector.ranking_
    indices = tuple(selector.get_support(indices=True))
    sorted_indices = sorted(indices, key=lambda i: scores[i], reverse=True)

    print("Selected features (in descending order of score):")
    print(sorted_indices)
    if feature_names is not None:
        print([feature_names[i] for i in sorted_indices])

    return features_transformed, indices


def variance_selection(feature_matrix, threshold, feature_names=None):
    """
    Variance threshold
    """
    selector = VarianceThreshold(threshold=threshold)
    transformed_features = selector.fit_transform(feature_matrix)
    print("Selected features shape:", transformed_features.shape)

    # Show variances
    variances = selector.variances_
    plt.plot(variances)
    plt.show()

    # Get indices of the selected features
    indices = tuple(selector.get_support(indices=True))
    print("Selected features indices:", indices)
    if feature_names is not None:
        print("Selected features names:", [feature_names[i] for i in indices])

    return transformed_features, indices


def person_correlation_selection(features, targets, features_to_select: int|float, feature_names=None):
    """
    Selects the features with the highest absolute correlation with the targets.
    :param features_to_select: if int, select the given number of features. If float, select the given percentage of features.
    """

    if isinstance(features_to_select, int):
        selector = SelectKBest(r_regression, k=features_to_select)
        features_transformed = selector.fit_transform(features, targets)
    elif isinstance(features_to_select, float):
        selector = SelectPercentile(r_regression, percentile=features_to_select)
        features_transformed = selector.fit_transform(features, targets)
    else:
        raise ValueError("'features_to_select' must be int or float")
    print("Selected features shape:", features_transformed.shape)

    # Show correlations
    scores = selector.scores_
    plt.scatter(range(len(feature_names)), scores)
    plt.show()

    # Get indices of the selected features
    indices = tuple(selector.get_support(indices=True))
    print("Selected features indices:", indices)
    if feature_names is not None:
        print("Selected features names:", [feature_names[i] for i in indices])

    return features_transformed, indices


def f_statistic_selection(features, targets, features_to_select: int|float, feature_names=None):
    """
    Selects the features with the highest linear F-test with the targets.
    :param features_to_select: if int, select the given number of features. If float, select the given percentage of features.
    """

    if isinstance(features_to_select, int):
        selector = SelectKBest(f_regression, k=features_to_select)
        features_transformed = selector.fit_transform(features, targets)
    elif isinstance(features_to_select, float):
        selector = SelectPercentile(f_regression, percentile=features_to_select)
        features_transformed = selector.fit_transform(features, targets)
    else:
        raise ValueError("'features_to_select' must be int or float")
    print("Selected features shape:", features_transformed.shape)

    # Show correlations
    scores = selector.scores_
    plt.scatter(range(len(feature_names)), scores)
    plt.show()

    # Get indices of the selected features
    indices = tuple(selector.get_support(indices=True))
    sorted_indices = sorted(indices, key=lambda i: scores[i], reverse=True)

    print("Selected features (in descending order of score):")
    print(sorted_indices)
    if feature_names is not None:
        print([feature_names[i] for i in sorted_indices])

    return features_transformed, indices


def mutual_information_selection(features, targets, features_to_select: int|float, feature_names=None):
    """
    Selects the features with the highest mutual information with the targets.
    :param features_to_select: if int, select the given number of features. If float, select the given percentage of features.
    """

    if isinstance(features_to_select, int):
        selector = SelectKBest(mutual_info_regression, k=features_to_select)
        features_transformed = selector.fit_transform(features, targets)
    elif isinstance(features_to_select, float):
        selector = SelectPercentile(mutual_info_regression, percentile=features_to_select)
        features_transformed = selector.fit_transform(features, targets)
    else:
        raise ValueError("'features_to_select' must be int or float")
    print("Selected features shape:", features_transformed.shape)

    # Show correlations
    scores = selector.scores_
    plt.scatter(range(len(feature_names)), scores)
    plt.show()

    # Get indices of the selected features
    indices = tuple(selector.get_support(indices=True))
    sorted_indices = sorted(indices, key=lambda i: scores[i], reverse=True)

    print("Selected features (in descending order of score):")
    print(sorted_indices)
    if feature_names is not None:
        print([feature_names[i] for i in sorted_indices])

    # Plot the selected features
    fig, ax = plt.subplots()
    ax.scatter(range(len(feature_names)), scores)
    ax.scatter(sorted_indices, [scores[i] for i in sorted_indices], color='red')
    ax.set_title("Mutual Information")
    ax.set_ylabel("Mutual Information")
    fig.tight_layout()
    plt.show()

    return features_transformed, indices


# 1) Read features
#features = read_all_features('Healthy Brain Network', multiples=True)
features = read_all_features('KJPP', multiples=True)
features.index = features.index.str.split('$').str[0]  # remove $ from the index
print("Features Shape before drop subjects:", features.shape)
features = features.dropna(axis=0)
print("Features Shape before drop features:", features.shape)
features = features.dropna(axis=1)
print("Features Shape:", features.shape)

# Discard bad-diagnoses
BAD_DIAGNOSES = np.loadtxt("/Volumes/MMIS-Saraiv/Datasets/KJPP/session_ids/bad_diagnoses.txt", dtype=str)
features = features.drop(BAD_DIAGNOSES, errors='ignore')
print("Features Shape after drop bad diagnoses:", features.shape)

# Shuffle columns order randomly
features = features.sample(frac=1, axis=1, random_state=0)

# 2) Read targets
#ages = read_ages('HBN')
ages = read_ages('KJPP')
targets = Series()
for index in features.index:
    if '$' in str(index):  # Multiples
        key = str(index).split('$')[0]  # remove the multiple
    else:  # Original
        key = index

    if key in ages:
        targets.loc[index] = ages[key]

targets = targets.dropna()  # Drop subject_sessions with nans targets
features = features.loc[targets.index]

print("Features Shape before drop wo/ages:", features.shape)

"""
# 4) Data Augmentation in the underrepresented MMSE scores

# round targets to integers
targets = targets.round().astype(int)

# Histogram before
plt.hist(targets, bins=17, rwidth=0.8)
plt.title("Before")
plt.show()

# 4.2. Data Augmentation method = SMOTE-C
#smote = SMOTE(random_state=42, k_neighbors=2, sampling_strategy='auto')
#features, targets = smote.fit_resample(features, targets)

# Histogram after
plt.hist(targets, bins=17, rwidth=0.8)
plt.title("After")
plt.show()

print("Features shape after DA:", features.shape)
"""

# 4) Convert features to an appropriate format
# e.g. {..., 'C9': (feature_names_C9, features_C9), 'C10': (feature_names_C10, features_C10), ...}
# to
# e.g. [..., features_C9, features_C10, ...]
feature_names = features.columns.to_numpy()
sessions = features.index.to_numpy()
features = [features.loc[code].to_numpy() for code in sessions]
dataset = []
for i, session in enumerate(sessions):
    dataset.append((features[i], targets[session]))


# 5) Separate 50% only for feature selection
#dataset, dataset_feature_selection = train_test_split(dataset, test_size=0.5, random_state=0, stratify=targets)
#print("Size of the dataset for feature selection:", len(dataset_feature_selection))

# 5.2) Define model
#model = GradientBoostingRegressor(n_estimators=300, max_depth=15, random_state=0, loss='absolute_error', learning_rate=0.04,)
model = RandomForestRegressor(n_estimators=300, random_state=0, )

# 6.1 Feature Selection (Pearson)
"""
objects = np.array([x[0] for x in dataset]) #dataset_feature_selection])
targets = np.array([x[1] for x in dataset]) #dataset_feature_selection])
transformed_features, indices = person_correlation_selection(objects, targets, feature_names=feature_names, features_to_select=200)
# update dataset
dataset = [([y for i, y in enumerate(x[0]) if i in indices], x[1]) for x in dataset]
print("Size of the dataset after Pearson feature selection:", len(dataset), len(dataset[0][0]))
"""
#"""
# 6.2 Feature Selection (RFE)
objects = np.array([x[0] for x in dataset]) #dataset_feature_selection])
targets = np.array([x[1] for x in dataset]) #dataset_feature_selection])
transformed_features, indices = rfe_selection(model, objects, targets, feature_names=feature_names, n_features_to_select=30, step=5)
# update dataset
dataset = [([y for i, y in enumerate(x[0]) if i in indices], x[1]) for x in dataset]
print("Size of the dataset after RFE selection:", len(dataset), len(dataset[0][0]))
#"""

