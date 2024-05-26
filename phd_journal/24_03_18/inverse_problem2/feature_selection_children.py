import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from pandas import Series
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.feature_selection import RFE, VarianceThreshold, SelectKBest, SelectPercentile, r_regression, f_regression, \
    mutual_info_regression, RFECV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split, cross_val_score

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


def cv_feature_selection(model, features, targets, cv, step, feature_names):
    """
    Recursive feature elimination with cross-validation
    """
    selector = RFECV(estimator=model, cv=cv, min_features_to_select=10, step=step, verbose=2, scoring='neg_mean_absolute_error',
                     n_jobs=-1)
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


def stratified_split(features, targets, val_size, random_state=0):
    """
    Returns stratified train and validation sets, by MMSE bins.
    """
    pass


# 1.1) Get all features
features, targets = read_children_cohorts()
print("Children features shape:", features.shape)

# Exclude examples age > 25 (these are errors)
features = features[targets <= 25]
targets = targets[targets <= 25]

# 3) Data Augmentation in the underrepresented MMSE scores
print("DATA AUGMENTATION")

# Dynamically define the age groups with bins of 3 years
G = 3
min_age, max_age = targets.min(), targets.max()
age_groups = np.arange(min_age, max_age, G)
# Get the number of samples in each group
age_distribution = [len(targets[(targets >= age) & (targets < age + G)]) for age in age_groups]
# Get majority score
max_samples = max(age_distribution)

print("Age distribution before augmentation:")
for i, age in enumerate(age_groups):
    print(f"Age group {age} -- {age+3}: {age_distribution[i]} examples")

# Augment all underrepresented scores up to the size of the majority score
for i, age in enumerate(age_groups):
    if age_distribution[i] < max_samples:
        # Get the number of samples to augment
        n_samples_to_augment = max_samples - age_distribution[i]
        # Get the samples to augment
        samples = features[targets.between(age, age + G)]
        # Augment with gaussian noise with sensitivity S
        S = 0.2
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

age_distribution_after = [len(targets[(targets >= age) & (targets < age + G)]) for age in age_groups]
#assert all([samples == max_samples for samples in age_distribution_after])
print("Age distribution after augmentation:")
for i, age in enumerate(age_groups):
    print(f"Age group {age} -- {age+3}: {age_distribution_after[i]} examples")


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


# 5) Separate 70% for feature selection + (CV) training  [STRATIFIED]
dataset_train, dataset_val = train_test_split(dataset, test_size=0.3, shuffle=True, random_state=0)
print("Size of 'train' dataset:", len(dataset_train))
print("Size of 'val' dataset:", len(dataset_val))

# 5.1) Plot histogram of ages of the train set and val set
plt.hist([x[1] for x in dataset_train], alpha=0.5, label='Feature Eng.', color='g')
plt.hist([x[1] for x in dataset_val], alpha=0.5, label='Validation', color='r')
plt.xlim(0, 25)
plt.legend(loc='upper right')
plt.show()


# 6.2) Define model
model = GradientBoostingRegressor(n_estimators=200, max_depth=10, random_state=0, loss='absolute_error',
                                  learning_rate=0.04,)

# 7. Feature Selection
print("Number of features:", len(dataset_train[0][0]))
objects = np.array([x[0] for x in dataset_train])
targets = np.array([x[1] for x in dataset_train])
transformed_features, indices = rfe_selection(model, objects, targets, n_features_to_select=80, feature_names=feature_names, step=5)

# 8. Get metrics for train set
model.fit(transformed_features, targets)
predictions = model.predict(transformed_features)
print('---------------------------------')
print('Train set metrics:')
mse = mean_squared_error(targets, predictions)
print(f'MSE: {mse}')
mae = mean_absolute_error(targets, predictions)
print(f'MAE: {mae}')
r2 = r2_score(targets, predictions)
print(f'R2: {r2}')
# 8.1. Plot regression between ground truth and predictions with seaborn and draw regression curve
plt.figure(figsize=((3.5,3.2)))
sns.regplot(x=targets, y=predictions, scatter_kws={'alpha': 0.4}, color="#34AC8B")
plt.xlabel('True MMSE (units)')
plt.ylabel('Predicted MMSE (units)')
plt.xlim(0, 30)
plt.ylim(0, 30)
plt.tight_layout()
plt.show()

#########
# VALIDATION

# 9. Update dataset_val

# Select features for validation set
dataset_val = [([y for i, y in enumerate(x[0]) if i in indices], x[1]) for x in dataset_val]
objects = np.array([x[0] for x in dataset_val])
targets = np.array([x[1] for x in dataset_val])

# 10. Make predictions
print("Size of validation dataset:", len(dataset_val))
print("Number of features:", len(dataset_val[0][0]))
predictions = model.predict(objects)

# 11. Get metrics for validation set
print('---------------------------------')
print('Validation set metrics:')
mse = mean_squared_error(targets, predictions)
print(f'MSE: {mse}')
mae = mean_absolute_error(targets, predictions)
print(f'MAE: {mae}')
r2 = r2_score(targets, predictions)
print(f'R2: {r2}')
# 8.1. Plot regression between ground truth and predictions with seaborn and draw regression curve
plt.figure(figsize=((3.5,3.2)))
sns.regplot(x=targets, y=predictions, scatter_kws={'alpha': 0.4}, color="#34AC8B")
plt.xlabel('True MMSE (units)')
plt.ylabel('Predicted MMSE (units)')
plt.xlim(0, 30)
plt.ylim(0, 30)
plt.tight_layout()
plt.show()


