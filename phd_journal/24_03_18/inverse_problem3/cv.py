from os import mkdir
from os.path import exists
from pickle import dump

import numpy as np
from math import floor, ceil
from matplotlib import pyplot as plt
import seaborn as sns
from pandas import Series
from seaborn import regplot
from sklearn.ensemble import GradientBoostingRegressor
from imblearn.over_sampling import SMOTE
#import ImbalancedLearningRegression as iblr
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import StratifiedKFold, KFold

#from pyloras import LORAS

from read import *
from read import read_all_features
from utils import feature_wise_normalisation, feature_wise_normalisation_with_coeffs


def augment(features, targets):
    # 4) Data Augmentation in the underrepresented MMSE scores

    # Histogram before
    plt.hist(targets, bins=27, rwidth=0.8)
    plt.title("Before")
    plt.show()

    # 4.1. Create more examples of missing targets, by interpolation of the existing ones
    def interpolate_missing_mmse(features, targets, missing_targets):
        #print("Missing targets:", missing_targets)
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
            lower_features.index = [str(l) + '_interpolated_' + str(u) for l, u in
                                    zip(lower_features_index, upper_features_index)]
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
            print(
                f"Interpolated {len(new_features)} examples for target {new_target}, from targets {lower_target} and {upper_target}")

            return features, targets

    while True:
        min_target = targets.min()
        max_target = targets.max()
        all_targets = targets.unique()
        missing_targets = [i for i in range(min_target, max_target + 1) if i not in all_targets]
        if len(missing_targets) == 0:
            break
        else:
            #print("New round of interpolation")
            features, targets = interpolate_missing_mmse(features, targets, missing_targets)

    # Histogram after interpolation
    plt.hist(targets, bins=27, rwidth=0.8)
    plt.title("After interpolation of missing targets")
    plt.show()

    # 4.2. Data Augmentation method = SMOTE-C
    for k in (5, 4, 3, 2, 1):
        try:
            smote = SMOTE(random_state=42, k_neighbors=k, sampling_strategy='auto')
            features, targets = smote.fit_resample(features, targets)
            print(f"Worked with k={k}")
            break
        except ValueError as e:
            print(f"Did not work with k={k}")

    # Histogram after
    plt.hist(targets, bins=27, rwidth=0.8)
    plt.title("After")
    plt.show()

    # Normalisation after DA
    features = feature_wise_normalisation(features, method='min-max')
    features = features.dropna(axis=1)

    print("Features shape after DA:", features.shape)
    return features, targets


def custom_cv(objects, targets, n_splits=5, random_state=42):
    """
    Custom Cross-Validation with Data Augmentation on-the-fly that ensures that the same subject is not present in both
    training and test sets.

    1. Identify the minority target and select 30% of its instances for the test set.
    2. Identify the INSIGHT examples in the test set and exclude other examples from the same subjects from the training set.
    3. Augment the remaining examples that can be selected for training.
    4. Select the remaining examples for training.

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
    # How much is the minority target?
    print(f"Minority target: {minority_target}")
    print(f"Minority target count: {np.min(counts)}")
    test_size = 1 / n_splits
    n_test_per_target = ceil(np.min(counts) * test_size)
    print(f"# test per target: {n_test_per_target}")

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
        print("Train examples before augmentation:", len(remaining_indices))
        augmented_objects, augmented_targets = augment(objects.iloc[remaining_indices], targets.iloc[remaining_indices])
        print("Train examples after augmentation:", len(augmented_objects))

        yield augmented_objects, objects.iloc[test_indices], augmented_targets, targets.iloc[test_indices]


def cv(model, objects, targets, folds: int, random_state:int):
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

    r2_scores = []
    mse_scores = []
    mae_scores = []

    for i, (train_objects, test_objects, train_targets, test_targets) in enumerate(custom_cv(objects, targets, n_splits=folds, random_state=random_state)):
        print(f"Fold {i+1}")

        # make sub-dir if not exists
        fold_path = join(out_path, str(i+1))
        if not exists(fold_path):
            mkdir(fold_path)

        # Train the model
        print(f"Train examples: {len(train_objects)}")
        model.fit(train_objects, train_targets)
        # save model
        with open(join(fold_path, "model.pkl"), 'wb') as f:
            dump(model, f)

        # Test the model
        print(f"Test examples: {len(test_objects)}")
        predictions = model.predict(test_objects)
        # save Dataframe predictions | targets of test set
        res = pd.DataFrame({'predictions': predictions, 'targets': test_targets})
        res.to_csv(join(out_path, 'predictions_targets.csv'))

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

        # Make regression plot
        plt.figure(figsize=(6, 5))
        plt.rcParams['font.family'] = 'Arial'
        sns.regplot(x=targets, y=predictions, scatter_kws={'alpha': 0.3, 'color': '#C60E4F'},
                    line_kws={'color': '#C60E4F'})
        plt.xlabel('True MMSE (units)', fontsize=12)
        plt.ylabel('Predicted MMSE (units)', fontsize=12)
        plt.xlim(-1.5, 31.5)
        plt.ylim(-1.5, 31.5)
        plt.xticks([0, 4, 6, 9, 12, 15, 20, 25, 30], fontsize=11)
        plt.yticks([0, 4, 6, 9, 12, 15, 20, 25, 30], fontsize=11)
        plt.grid(linestyle='--', alpha=0.4)
        plt.box(False)
        plt.tight_layout()
        plt.savefig(join(out_path, 'train.png'))
    # Print the average scores
    print(f'Average R2: {np.mean(r2_scores)} +/- {np.std(r2_scores)}')
    print(f'Average MSE: {np.mean(mse_scores)} +/- {np.std(mse_scores)}')
    print(f'Average MAE: {np.mean(mae_scores)} +/- {np.std(mae_scores)}')


out_path = './scheme57/cv'
model_path = './scheme57'

FEATURES_SELECTED = ['Hjorth#Complexity#T5', 'Hjorth#Complexity#F4',
                     'COH#Frontal(R)-Parietal(L)#delta', 'Hjorth#Complexity#T3',
                     'Spectral#RelativePower#F7#theta', 'COH#Frontal(R)-Temporal(L)#theta',
                     'Spectral#EdgeFrequency#O2#beta', 'COH#Frontal(L)-Temporal(R)#beta',
                     'COH#Temporal(L)-Parietal(L)#gamma', 'Spectral#EdgeFrequency#O1#beta',
                     'COH#Frontal(R)-Parietal(L)#theta', 'COH#Temporal(L)-Temporal(R)#alpha',
                     'COH#Frontal(R)-Temporal(L)#gamma', 'COH#Temporal(R)-Parietal(L)#beta',
                     'COH#Frontal(R)-Occipital(L)#theta', 'COH#Temporal(L)-Parietal(L)#beta',
                     'Hjorth#Activity#F7', 'COH#Occipital(L)-Occipital(R)#gamma',
                     'Spectral#Flatness#P3#beta', 'COH#Temporal(R)-Parietal(R)#alpha',
                     'Spectral#Entropy#P3#alpha', 'COH#Frontal(R)-Parietal(R)#theta',
                     'COH#Frontal(R)-Temporal(L)#delta', 'Spectral#Entropy#O2#alpha',
                     'Spectral#Entropy#T4#theta', 'Spectral#RelativePower#Cz#beta',
                     'Spectral#Diff#Pz#delta', 'COH#Parietal(R)-Occipital(L)#beta',
                     'Spectral#EdgeFrequency#Fz#beta', 'Spectral#Diff#Cz#gamma',
                     'Spectral#RelativePower#Fp1#gamma', 'COH#Frontal(R)-Parietal(L)#gamma',
                     'PLI#Frontal(R)-Parietal(L)#alpha', 'Spectral#Diff#F7#beta',
                     'Hjorth#Mobility#O1', 'Spectral#Flatness#T4#gamma',
                     'PLI#Parietal(L)-Occipital(L)#gamma', 'Spectral#Flatness#T6#delta',
                     'COH#Parietal(R)-Occipital(L)#alpha',
                     'COH#Parietal(R)-Occipital(R)#beta', 'Spectral#Diff#T4#delta',
                     'Spectral#Diff#F8#alpha', 'COH#Temporal(R)-Occipital(L)#beta',
                     'COH#Parietal(R)-Occipital(L)#gamma', 'Hjorth#Mobility#P4',
                     'COH#Frontal(L)-Temporal(L)#beta',
                     'COH#Occipital(L)-Occipital(R)#alpha', 'Spectral#Entropy#T3#theta',
                     'COH#Frontal(R)-Occipital(R)#alpha', 'Hjorth#Complexity#P3',
                     'COH#Frontal(L)-Occipital(L)#beta', 'Hjorth#Activity#C3',
                     'COH#Temporal(L)-Occipital(R)#theta', 'Spectral#Diff#F4#beta',
                     'COH#Frontal(L)-Frontal(R)#gamma', 'Spectral#Diff#C3#gamma',
                     'COH#Frontal(L)-Frontal(R)#theta', 'COH#Parietal(L)-Occipital(R)#theta',
                     'Spectral#RelativePower#F7#gamma', 'Spectral#RelativePower#F3#beta',
                     'PLI#Temporal(R)-Parietal(R)#beta', 'Spectral#Flatness#F7#beta',
                     'Hjorth#Complexity#O2', 'Spectral#Entropy#Cz#theta',
                     'PLI#Frontal(R)-Occipital(R)#beta', 'COH#Temporal(L)-Parietal(R)#beta',
                     'COH#Frontal(L)-Occipital(L)#delta', 'Spectral#Flatness#F8#delta',
                     'Spectral#Entropy#F4#delta', 'PLI#Temporal(R)-Parietal(R)#gamma',
                     'COH#Occipital(L)-Occipital(R)#delta',
                     'COH#Temporal(L)-Parietal(R)#delta', 'PLI#Frontal(L)-Temporal(R)#delta',
                     'Spectral#Flatness#P3#theta', 'Spectral#Entropy#F7#alpha',
                     'COH#Frontal(R)-Temporal(R)#delta', 'COH#Frontal(L)-Occipital(R)#gamma',
                     'COH#Frontal(L)-Frontal(R)#beta', 'Hjorth#Complexity#Cz',
                     'COH#Frontal(L)-Occipital(R)#beta']

# 1) Read features
# 1.1. Multiples = yes
# 1.2. Which multiples = independent ones
# 1.3. Which features = FEATURES_SELECTED
miltiadous = read_all_features('Miltiadous Dataset', multiples=True)
brainlat = read_all_features('BrainLat', multiples=True)
sapienza = read_all_features('Sapienza', multiples=True)
insight = read_all_features('INSIGHT', multiples=True)
features = pd.concat([brainlat, miltiadous, sapienza, insight], axis=0)
features = features[FEATURES_SELECTED]
features = features.dropna(axis=0)
print("Features Shape:", features.shape)

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

# 3) Normalisation of all dataset
features = feature_wise_normalisation(features, method='min-max')
features = features.dropna(axis=1)

# 4) Define model
model = GradientBoostingRegressor(n_estimators=300, max_depth=15, random_state=0, loss='absolute_error',
                                  learning_rate=0.04, )

# 5) Cross-Validation
cv(model, features, targets, 5,42)
