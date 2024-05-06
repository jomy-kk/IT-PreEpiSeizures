import numpy as np
from matplotlib import pyplot as plt
from pandas import Series
from seaborn import regplot
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split, cross_val_score, KFold, StratifiedKFold

from read import *
from read import read_all_features
from utils import feature_wise_normalisation


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


def augment(objects, targets):
    """
    Augment the objects set by adding Gaussian noise to the feature vectors, up to the size of the majority target.
    The same target is assigned to the augmented feature vector.
    Args:
        objects: A DataFrame of feature vectors
        targets: A Series array of target values

    Returns:
        The augmented objects and targets.
    """

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
            samples = objects[targets == score]
            # Augment with gaussian noise with sensitivity S
            S = 0.1
            i = 0
            n_cycles = 1
            while n_samples_to_augment > 0:
                # Augment
                augmented = samples.iloc[i] + np.random.normal(0, S, len(samples.columns))
                name = str(samples.index[i]) + '_augmented_' + str(n_cycles)
                # Append
                objects.loc[name] = augmented
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

    return objects, targets


def custom_cv(objects, targets, n_splits=5, test_size=0.3):
    """
    Custom Cross-Validation with Data Augmentation on-the-fly that ensures that the same subject is not present in both
    training and test sets.

    1. Identify the minority target and select 30% of its instances for the test set.
I   2. Identify the INSIGHT examples in the test set and exclude other examples from the same subjects from the training set.
A   3. Augment the remaining examples that can be selected for training.
S   4. Select the remaining examples for training.

    Args:
        objects: A DataFrame of feature vectors
        targets: A Series of target values
        test_size: The proportion of the minority target to be selected for the test set

    Returns:
        The augmented training objects, test objects, training targets, and test targets.

    """

    # Create a list of indices
    indices = np.arange(len(targets))

    # Create a list to keep track of the indices that have already been used for the test set
    used_test_indices = []

    # Identify the minority target
    unique_targets, counts = np.unique(targets, return_counts=True)
    minority_target = unique_targets[np.argmin(counts)]
    # How much is 30% of the minority target?
    print(f"Minority target: {minority_target}")
    print(f"Minority target count: {np.min(counts)}")
    n_test_per_target = 4 #int(np.min(counts) * test_size)  FIXME: hardcoded

    # for each fold...
    for i in range(n_splits):
        # 1. Select 'n_test_per_target' per each target to constitute the test set
        test_indices = []
        for target in unique_targets:

            # Get the indices for this target that have not been used for the test set yet
            target_indices = [index for index in indices[targets == target] if index not in used_test_indices]

            if len(target_indices) < n_test_per_target:
                continue  # If the number of examples of this target is less than 'n_test_per_target', do not select any; they will be used in the training set
            Y = np.random.choice(target_indices, size=n_test_per_target, replace=False)
            test_indices.extend(Y)
        # Add the test indices for this fold to the list of used test indices
        used_test_indices.extend(test_indices)

        # Print targets distribution for test set
        print(f"Test set distribution:")
        test_dist = {target: len([i for i in test_indices if targets.iloc[i] == target]) for target in unique_targets}
        print(test_dist)


        # 2. Identify the INSIGHT examples in the test set, and find remaining examples for training
        insight_indices = [i for i in test_indices if '_' in str(objects.index[i])]
        # Exclude other examples from the same subjects from the training set
        excluded_subjects = {objects.index[i].split('_')[0] for i in insight_indices if '_' in objects.index[i]}
        remaining_indices = [i for i in range(len(targets)) if i not in test_indices and str(objects.index[i]).split('_')[0] not in excluded_subjects]

        # Print amount unused examples
        used = len(remaining_indices) + len(test_indices)
        print(f"Unused examples: {len(targets) - used}")

        # 3. Augment the remaining examples that can be selected for training
        augmented_objects, augmented_targets = augment(objects.iloc[remaining_indices], targets.iloc[remaining_indices])

        yield augmented_objects, objects.iloc[test_indices], augmented_targets, targets.iloc[test_indices]


def cv(model, objects, targets, folds: int, stratified:bool, augmentation: bool, shuffle:bool, random_state:int):
    """
    My own Implementation of Cross-Validation with Data Augmentation on-the-fly that ensures that the same subject
    is not present in both training and test sets.
    Args:
        model: A Sklearn model
        objects: The feature vectors of each example in DataFrame format
        targets: The target values of each example in Series format
        folds: The number of folds, k
        stratified: Whether to use stratified k-fold
        shuffle: Whether to shuffle the data before splitting
        random_state: Random seed for reproducibility, if shuffle is True

    Prints:
        The average R2, MSE, and MAE scores across all folds.

    Plots:
        The regression plot between the true and predicted MMSE scores for each fold.
    """

    if stratified:
        cv = StratifiedKFold(n_splits=folds, shuffle=shuffle, random_state=random_state)
    else:
        cv = KFold(n_splits=folds, shuffle=shuffle, random_state=random_state)

    r2_scores = []
    mse_scores = []
    mae_scores = []

    #for i, (train_index, test_index) in enumerate(cv.split(objects, targets)):
    for i, (train_objects, test_objects, train_targets, test_targets) in enumerate(custom_cv(objects, targets, n_splits=5)):
        print(f"Fold {i+1}")

        # Train the model
        model.fit(train_objects, train_targets)
        print(f"Train examples: {len(train_objects)}")

        # Test the model
        predictions = model.predict(test_objects)
        print(f"Test examples: {len(test_objects)}")

        # Calculate the scores
        r2 = r2_score(test_targets, predictions)
        print(f"R2: {r2}")
        mse = mean_squared_error(test_targets, predictions)
        print(f"MSE: {mse}")
        mae = mean_absolute_error(test_targets, predictions)
        print(f"MAE: {mae}")

        # Append the scores
        r2_scores.append(r2)
        mse_scores.append(mse)
        mae_scores.append(mae)

        # Plot the regression plot
        plt.figure(figsize=((3.5,3.2)))
        plt.title(f"Fold {i+1}")
        regplot(x=test_targets, y=predictions, scatter_kws={'alpha': 0.4, 's':20}, color="#C60E4F")
        plt.xlabel('True MMSE (units)')
        plt.ylabel('Predicted MMSE (units)')
        plt.xlim(0, 30)
        plt.ylim(0, 30)
        plt.tight_layout()
        plt.show()

    # Print the average scores
    print(f'Average R2: {np.mean(r2_scores)} +/- {np.std(r2_scores)}')
    print(f'Average MSE: {np.mean(mse_scores)} +/- {np.std(mse_scores)}')
    print(f'Average MAE: {np.mean(mae_scores)} +/- {np.std(mae_scores)}')


# 1. Read features

FEATURES_SELECTED = ['Spectral#RelativePower#C3#beta1', 'Spectral#EdgeFrequency#C3#beta3', 'Spectral#RelativePower#C3#gamma', 'Spectral#EdgeFrequency#C4#alpha1', 'Spectral#RelativePower#C4#beta3', 'Spectral#EdgeFrequency#C4#beta3', 'Spectral#EdgeFrequency#C4#gamma', 'Spectral#Flatness#Cz#theta', 'Spectral#PeakFrequency#Cz#theta', 'Spectral#EdgeFrequency#Cz#beta3', 'Spectral#EdgeFrequency#Cz#gamma', 'Spectral#PeakFrequency#Cz#gamma', 'Spectral#RelativePower#F3#beta1', 'Spectral#Diff#F4#delta', 'Spectral#RelativePower#F7#beta3', 'Spectral#EdgeFrequency#F7#beta3', 'Spectral#RelativePower#F7#gamma', 'Spectral#RelativePower#F8#beta1', 'Spectral#EdgeFrequency#F8#beta3', 'Spectral#RelativePower#Fp1#beta1', 'Spectral#EdgeFrequency#Fp1#beta3', 'Spectral#Diff#Fp2#delta', 'Spectral#RelativePower#Fp2#beta1', 'Spectral#RelativePower#Fp2#beta3', 'Spectral#Diff#Fpz#beta2', 'Spectral#Entropy#O1#delta', 'Spectral#RelativePower#O1#beta2', 'Spectral#EdgeFrequency#O1#beta2', 'Spectral#EdgeFrequency#O1#beta3', 'Spectral#RelativePower#O2#delta', 'Spectral#PeakFrequency#O2#alpha1', 'Spectral#RelativePower#O2#beta1', 'Spectral#RelativePower#O2#beta3', 'Spectral#Diff#P3#beta1', 'Spectral#RelativePower#P3#beta3', 'Spectral#RelativePower#Pz#alpha1', 'Spectral#EdgeFrequency#Pz#beta3', 'Spectral#RelativePower#T4#alpha1', 'Spectral#RelativePower#T4#beta3', 'Spectral#RelativePower#T4#gamma', 'Spectral#EdgeFrequency#T5#beta2', 'Hjorth#Complexity#T5', 'Hjorth#Complexity#P4', 'Hjorth#Complexity#F7', 'Hjorth#Complexity#T4', 'Hjorth#Complexity#F8', 'Hjorth#Complexity#T3', 'Hjorth#Mobility#P3', 'PLI#Frontal(L)-Temporal(R)#alpha1', 'PLI#Frontal(L)-Occipital(L)#alpha1', 'PLI#Frontal(R)-Temporal(R)#alpha1', 'PLI#Temporal(R)-Parietal(R)#alpha1', 'PLI#Temporal(R)-Occipital(L)#alpha1', 'PLI#Parietal(R)-Occipital(L)#alpha1', 'PLI#Occipital(L)-Occipital(R)#alpha1', 'PLI#Temporal(R)-Occipital(R)#alpha2', 'PLI#Parietal(R)-Occipital(L)#alpha2', 'COH#Frontal(L)-Frontal(R)#theta', 'COH#Frontal(L)-Occipital(L)#theta', 'COH#Frontal(L)-Occipital(R)#alpha1', 'COH#Frontal(R)-Occipital(L)#alpha1', 'COH#Parietal(R)-Occipital(L)#alpha1', 'COH#Frontal(L)-Frontal(R)#alpha2', 'COH#Frontal(L)-Occipital(R)#alpha2', 'COH#Parietal(R)-Occipital(L)#alpha2', 'COH#Parietal(R)-Occipital(R)#alpha2', 'COH#Occipital(L)-Occipital(R)#alpha2', 'COH#Frontal(L)-Occipital(L)#beta1', 'COH#Temporal(R)-Parietal(R)#beta1', 'COH#Parietal(R)-Occipital(R)#beta1', 'COH#Frontal(L)-Parietal(L)#beta2', 'COH#Frontal(R)-Occipital(L)#beta2', 'COH#Frontal(L)-Temporal(R)#beta3', 'COH#Frontal(L)-Parietal(L)#beta3', 'COH#Frontal(L)-Occipital(L)#beta3', 'COH#Frontal(L)-Occipital(R)#beta3', 'COH#Frontal(R)-Occipital(L)#beta3', 'COH#Temporal(L)-Occipital(R)#beta3', 'COH#Frontal(L)-Occipital(R)#gamma', 'COH#Frontal(R)-Occipital(R)#gamma']

# 1.1) Get selected features
insight = read_all_features('INSIGHT')
insight = insight[FEATURES_SELECTED]
print("INSIGHT shape (all):", insight.shape)
insight_before = insight.shape[0]
insight = insight.dropna(axis=0)  # drop sessions with missing values
insight_after = insight.shape[0]
print("INSIGHT shape (sessions w/ required features):", insight.shape, f"({insight_before - insight_after} sessions dropped)")

brainlat = read_all_features('BrainLat')
brainlat = brainlat[FEATURES_SELECTED]
print("BrainLat shape (all):", brainlat.shape)
brainlat_before = brainlat.shape[0]
brainlat = brainlat.dropna(axis=0)  # drop sessions with missing values
brainlat_after = brainlat.shape[0]
print("BrainLat shape (sessions w/ required features):", brainlat.shape, f"({brainlat_before - brainlat_after} sessions dropped)")

miltiadous = read_all_features('Miltiadous Dataset')
miltiadous = miltiadous[FEATURES_SELECTED]
print("Miltiadous shape (all):", miltiadous.shape)
miltiadous_before = miltiadous.shape[0]
miltiadous = miltiadous.dropna(axis=0)  # drop sessions with missing values
miltiadous_after = miltiadous.shape[0]
print("Miltiadous shape (sessions w/ required features):", miltiadous.shape, f"({miltiadous_before - miltiadous_after} sessions dropped)")

# EXTRA: Read multiples examples (from fake subjects)
multiples = read_all_features_multiples()
multiples = multiples[FEATURES_SELECTED]
print("Multiples shape:", multiples.shape)
multiples_before = multiples.shape[0]
multiples = multiples.dropna(axis=0)  # drop sessions with missing values
multiples_after = multiples.shape[0]
print("Multiples shape (sessions w/ required features):", multiples.shape, f"({multiples_before - multiples_after} sessions dropped)")

# Perturb the multiple features, so they are not identical to the original ones
# These sigma values were defined based on similarity with the original features; the goal is to make them disimilar inasmuch as other examples from other subjects.
jitter = lambda x: x + np.random.normal(0, 0.1, x.shape)
scaling = lambda x: x * np.random.normal(1, 0.04, x.shape)
print("Perturbing multiple examples...")
for feature in multiples.columns:
    data = multiples[feature].values
    data = jitter(data)
    data = scaling(data)
    multiples[feature] = data

features = pd.concat([insight, brainlat, miltiadous, multiples], axis=0)
print("Read all features. Final Shape:", features.shape)
print(f"Discarded a total of {insight_before - insight_after + brainlat_before - brainlat_after + miltiadous_before - miltiadous_after} sessions with missing values.")

# 1.2) Normalise feature vectors
features = feature_wise_normalisation(features, method='min-max')
features = features.dropna(axis=1)

# 2) Read targets
insight_targets = read_mmse('INSIGHT')
brainlat_targets = read_mmse('BrainLat')
miltiadous_targets = read_mmse('Miltiadous Dataset')
targets = Series()
for index in features.index:
    if '_' in str(index):  # insight
        key = int(index.split('_')[0])
        if key in insight_targets:
            targets.loc[index] = insight_targets[key]
    elif '-' in str(index):  # brainlat
        if index in brainlat_targets:
            targets.loc[index] = brainlat_targets[index]
    else:  # miltiadous
        # parse e.g. 24 -> 'sub-024'; 1 -> 'sub-001'
        if '$' in str(index):  # EXTRA: multiple examples, remove the $ and the number after it; the target is the same
            key = 'sub-' + str(str(index).split('$')[0]).zfill(3)
        else:
            key = 'sub-' + str(index).zfill(3)
        if key:
            targets.loc[index] = miltiadous_targets[key]

print("Read targets. Shape:", targets.shape)

# Drop subject_sessions with nans targets
targets = targets.dropna()
features_sessions_before = set(features.index)
features = features.loc[targets.index]
features_sessions_after = set(features.index)
print("After Dropping sessions with no targets - Shape:", features.shape)
print("Dropped sessions:", features_sessions_before - features_sessions_after)

# 5) Define model
model = GradientBoostingRegressor(n_estimators=200, max_depth=10, random_state=0, loss='absolute_error',
                                  learning_rate=0.04,)

# 6) Cross-validation
cv(model, features, targets, folds=5, stratified=True, augmentation=True, shuffle=True, random_state=0)
