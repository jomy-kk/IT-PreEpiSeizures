import numpy as np
from imblearn.over_sampling import SMOTE
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

# 1) Read features
# 1.1. Multiples = yes
# 1.2. Which multiples = all (bc in RFE there's no test)
# 1.3. Which features = FEATURES_SELECTED
miltiadous = read_all_features('Miltiadous Dataset', multiples=True)
brainlat = read_all_features('BrainLat', multiples=True)
sapienza = read_all_features('Sapienza', multiples=True)
insight = read_all_features('INSIGHT', multiples=True)
features = pd.concat([brainlat, miltiadous, sapienza, insight], axis=0)
features = features.dropna(axis=1)
print("Features Shape:", features.shape)

# Shuffle columns order randomly
features = features.sample(frac=1, axis=1, random_state=0)

# 2) Read targets
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
targets = targets.dropna()  # Drop subject_sessions with nans targets
features = features.loc[targets.index]

# EXTRA:
# Discard 500 examples of MMSE 30
targets_30 = targets[targets == 30]
targets_30 = targets_30.sample(n=500, random_state=0)
targets = targets.drop(targets_30.index)
features = features.loc[targets.index]

# 4) Data Augmentation in the underrepresented MMSE scores

# Histogram before
plt.hist(targets, bins=27, rwidth=0.8)
plt.title("Before")
plt.show()

# 4.0. Create more examples of missing targets, by interpolation of the existing ones
def interpolate_missing_mmse(features, targets, missing_targets):
    print("Missing targets:", missing_targets)
    for target in missing_targets:
        # Find the closest targets
        lower_target = max([t for t in targets if t < target])
        upper_target = min([t for t in targets if t > target])
        # Get the features of the closest targets
        lower_features = features[targets == lower_target]
        upper_features = features[targets == upper_target]
        # make them the same size
        n_lower = len(lower_features)
        n_upper = len(upper_features)
        if n_lower > n_upper:
            lower_features = lower_features.sample(n_upper)
        elif n_upper > n_lower:
            upper_features = upper_features.sample(n_lower)
        else:
            pass
        # Change index names
        # upper.index is [a, b, c, d, ...]
        # lower.index is [e, f, g, h, ...]
        # final index [a_interpolated_e, b_interpolated_f, c_interpolated_g, d_interpolated_h, ...]
        lower_features_index, upper_features_index = upper_features.index, lower_features.index
        lower_features.index = [str(l) + '_interpolated_' + str(u) for l, u in zip(lower_features_index, upper_features_index)]
        upper_features.index = lower_features.index

        # Interpolate
        new_features = (lower_features + upper_features) / 2
        # has this nans?
        if new_features.isnull().values.any():
            print("Nans in the interpolated features")
            exit(-2)
        # Append
        features = pd.concat([features, new_features])
        new_target = int((lower_target + upper_target) / 2)
        targets = pd.concat([targets, Series([new_target] * len(new_features), index=new_features.index)])
        print(f"Interpolated {len(new_features)} examples for target {new_target}, from targets {lower_target} and {upper_target}")

        return features, targets

while True:
    min_target = targets.min()
    max_target = targets.max()
    all_targets = targets.unique()
    missing_targets = [i for i in range(min_target, max_target + 1) if i not in all_targets]
    if len(missing_targets) == 0:
        break
    else:
        print("New round of interpolation")
        features, targets = interpolate_missing_mmse(features, targets, missing_targets)

# Histogram after interpolation
plt.hist(targets, bins=27, rwidth=0.8)
plt.title("After interpolation of missing targets")
plt.show()

# 4.2. Data Augmentation method = SMOTE-C
smote = SMOTE(random_state=42, k_neighbors=5, sampling_strategy='auto')
features, targets = smote.fit_resample(features, targets)

# Histogram after
plt.hist(targets, bins=27, rwidth=0.8)
plt.title("After")
plt.show()

print("Features shape after DA:", features.shape)


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
dataset, dataset_feature_selection = train_test_split(dataset, test_size=0.5, random_state=0, stratify=targets)
print("Size of the dataset for feature selection:", len(dataset_feature_selection))

# 5.2) Define model
model = GradientBoostingRegressor(n_estimators=300, max_depth=15, random_state=0, loss='absolute_error',
                                  learning_rate=0.04,)

# 6. Feature Selection (RFE)
objects = np.array([x[0] for x in dataset_feature_selection])
targets = np.array([x[1] for x in dataset_feature_selection])
transformed_features, indices = rfe_selection(model, objects, targets, feature_names=feature_names, n_features_to_select=80, step=5)

# update dataset for training and testing
dataset = [([y for i, y in enumerate(x[0]) if i in indices], x[1]) for x in dataset]

# print the selected features names
print("Selected features:")
print()

# 5. Train and Test (without CV)
print("Size of the dataset:", len(dataset))
print("Number of features:", len(dataset[0][0]))
objects = np.array([x[0] for x in dataset])
targets = np.array([x[1] for x in dataset])
# Split training and testing
objects_train, objects_test, targets_train, targets_test = train_test_split(objects, targets, test_size=0.3, random_state=0, shuffle=True, stratify=targets)
# Train
model.fit(objects_train, targets_train)
# Test
preditcions = model.predict(objects_test)
# MAE
mae = np.mean(np.abs(targets_test - preditcions))
print("MAE:", mae)
# MSE
mse = np.mean((targets_test - preditcions) ** 2)
print("MSE:", mse)
# R2
r2 = model.score(objects_test, targets_test)
print("R2:", r2)

# 6.1. Feature Importances
feature_importances(model)

