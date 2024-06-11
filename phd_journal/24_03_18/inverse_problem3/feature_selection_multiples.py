import numpy as np
from matplotlib import pyplot as plt
from pandas import Series
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.feature_selection import RFE, VarianceThreshold, SelectKBest, SelectPercentile, r_regression, f_regression, \
    mutual_info_regression
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


def feature_importances(model):
    importances = model.feature_importances_
    importances = pd.Series(importances, index=feature_names)
    importances = importances.nlargest(20) # Get max 20 features
    fig, ax = plt.subplots()
    importances.plot.bar(ax=ax)
    ax.set_title("Feature importances using MDI")
    ax.set_ylabel("Mean decrease in impurity")
    fig.tight_layout()
    plt.show()


def train_test_cv(model, cv, objects, targets):
    scores = cross_val_score(model, objects, targets,
                             cv=cv, scoring='r2', #'neg_mean_absolute_error',
                             verbose=2, n_jobs=-1)
    print("Cross-Validation mean score:", scores.mean())
    print("Cross-Validation std score:", scores.std())
    print("Cross-Validation max score:", scores.max())
    print("Cross-Validation min score:", scores.min())


# 1.1) Get all features
miltiadous = read_all_features('Miltiadous Dataset', multiples=True)
brainlat = read_all_features('BrainLat', multiples=True)
sapienza = read_all_features('Sapienza', multiples=True)
insight = read_all_features('INSIGHT', multiples=True)
features = pd.concat([brainlat, miltiadous, sapienza, insight], axis=0)
print("Read all features. Shape:", features.shape)

# 2) Get targets
insight_targets = read_mmse('INSIGHT')
brainlat_targets = read_mmse('BrainLat')
miltiadous_targets = read_mmse('Miltiadous Dataset')
sapienza_targets = read_mmse('Sapienza')
targets = Series()
for index in features.index:
    if '$' in str(index):  # Multiples
        key = str(index).split('$')[0]  # remove the multiple
    else:  # Original
        key = index

    if '_' in str(key):  # insight
        key = int(key.split('_')[0])
        if key in insight_targets:
            targets.loc[index] = insight_targets[key]
    elif '-' in str(key):  # brainlat
        if key in brainlat_targets:
            targets.loc[index] = brainlat_targets[key]
    elif 'PARTICIPANT' in str(key):  # sapienza
        if key in sapienza_targets:
            targets.loc[index] = sapienza_targets[key]
    else:  # miltiadous
        # parse e.g. 24 -> 'sub-024'; 1 -> 'sub-001'
        key = 'sub-' + str(key).zfill(3)
        if key:
            targets.loc[index] = miltiadous_targets[key]

# Drop subject_sessions with nans targets
targets = targets.dropna()
features = features.loc[targets.index]

# Normalise feature vectors BEFORE
features = feature_wise_normalisation(features, method='min-max')
features = features.dropna(axis=1)


######################
# 3) Data Augmentation in the underrepresented MMSE scores
print("DATA AUGMENTATION")

# Dynamically define the MMSE groups with bins of 2 MMSE scores
mmse_scores = sorted(list(set(targets)))
# Get the number of samples in each group
mmse_distribution = [len(targets[targets == mmse]) for mmse in mmse_scores]
# Get majority score
max_samples = max(mmse_distribution)

print("MMSE distribution before augmentation:")
for i, mmse in enumerate(mmse_scores):
    print(f"MMSE {mmse}: {mmse_distribution[i]} examples")

# Augment all underrepresented scores up to the size of the majority score
for i, score in enumerate(mmse_scores):
    if mmse_distribution[i] < max_samples:
        # Get the number of samples to augment
        n_samples_to_augment = max_samples - mmse_distribution[i]
        # Get the samples to augment
        samples = features[targets == score]
        # Augment with gaussian noise with sensitivity S
        S = 0.1
        i = 0
        n_cycles = 1
        while n_samples_to_augment > 0:
            # Augment
            augmented = samples.iloc[i] + np.random.normal(0, S, len(samples.columns))
            name = str(samples.index[i]) + '_augmented_' + str(n_cycles)
            # Append
            features.loc[name] = augmented
            targets.loc[name] = targets[samples.index[i]]
            # Update
            i += 1
            n_samples_to_augment -= 1
            if i == len(samples):
                i = 0
                n_cycles += 1

mmse_distribution_after = [len(targets[targets == mmse]) for mmse in mmse_scores]
assert all([samples == max_samples for samples in mmse_distribution_after])
print("MMSE distribution after augmentation:")
for i, mmse in enumerate(mmse_scores):
    print(f"MMSE {mmse}: {mmse_distribution_after[i]} examples")
######################

# Normalise feature vectors AFTER
features = feature_wise_normalisation(features, method='min-max')
features = features.dropna(axis=1)


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


# 5) Separate 70% only for feature selection
dataset, dataset_feature_selection = train_test_split(dataset, test_size=0.7, random_state=0)
print("Size of the dataset for feature selection:", len(dataset_feature_selection))

# 5.2) Define model
model = GradientBoostingRegressor(n_estimators=300, max_depth=15, random_state=0, loss='absolute_error',
                                  learning_rate=0.04,)

# 6.1. Feature Selection 1 (Pearson Correlation)
print("Number of features before:", len(dataset_feature_selection[0][0]))
objects = np.array([x[0] for x in dataset_feature_selection])
targets = np.array([x[1] for x in dataset_feature_selection])
transformed_features, indices = person_correlation_selection(objects, targets, features_to_select=200, feature_names=feature_names)
print("Number of features after Pearson exclusion:", transformed_features.shape[1])
# update dataset for further feature selection
dataset_feature_selection = [([y for i, y in enumerate(x[0]) if i in indices], x[1]) for x in dataset_feature_selection]
# update dataset for training and testing
dataset = [([y for i, y in enumerate(x[0]) if i in indices], x[1]) for x in dataset]

# 6.2. Feature Selection 2 (RFE)
objects = np.array([x[0] for x in dataset_feature_selection])
targets = np.array([x[1] for x in dataset_feature_selection])
# different methods
transformed_features, indices = rfe_selection(model, objects, targets, feature_names=feature_names, n_features_to_select=80, step=5)

# update dataset for training and testing
dataset = [([y for i, y in enumerate(x[0]) if i in indices], x[1]) for x in dataset]

# print the selected features names
print("Selected features:")
print([feature_names[i] for i in indices])

# 5. Train and Test
print("Size of the dataset:", len(dataset))
print("Number of features:", len(dataset[0][0]))
objects = np.array([x[0] for x in dataset])
targets = np.array([x[1] for x in dataset])
cv = KFold(5, shuffle=True)  # leave 20% out, non-overlapping test sets
train_test_cv(model, cv, objects, targets)


