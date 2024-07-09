from os import mkdir
from os.path import exists
from pickle import dump

import numpy as np
from matplotlib import pyplot as plt
from pandas import Series
from seaborn import regplot
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split, cross_val_score, KFold

from read import *
from read import read_all_features


FEATURES_SELECTED = ['Spectral#Flatness#C3#delta', 'Spectral#PeakFrequency#C3#delta', 'Spectral#Entropy#C3#theta', 'Spectral#PeakFrequency#C3#theta', 'Spectral#RelativePower#C3#alpha', 'Spectral#Entropy#C3#alpha', 'Spectral#RelativePower#C3#beta', 'Spectral#Diff#C3#beta', 'Spectral#EdgeFrequency#C3#gamma', 'Spectral#RelativePower#C4#delta', 'Spectral#Entropy#C4#delta', 'Spectral#EdgeFrequency#C4#delta', 'Spectral#RelativePower#C4#theta', 'Spectral#Flatness#C4#theta', 'Spectral#PeakFrequency#C4#theta', 'Spectral#Entropy#C4#alpha', 'Spectral#RelativePower#C4#beta', 'Spectral#Entropy#C4#beta', 'Spectral#EdgeFrequency#C4#beta', 'Spectral#Diff#C4#beta', 'Spectral#RelativePower#Cz#delta', 'Spectral#Entropy#Cz#delta', 'Spectral#Diff#Cz#delta', 'Spectral#EdgeFrequency#Cz#theta', 'Spectral#Diff#Cz#theta', 'Spectral#Entropy#Cz#alpha', 'Spectral#EdgeFrequency#Cz#beta', 'Spectral#Diff#Cz#beta', 'Spectral#Flatness#Cz#gamma', 'Spectral#PeakFrequency#F3#delta', 'Spectral#RelativePower#F3#theta', 'Spectral#EdgeFrequency#F3#theta', 'Spectral#RelativePower#F3#alpha', 'Spectral#Entropy#F3#alpha', 'Spectral#EdgeFrequency#F3#alpha', 'Spectral#EdgeFrequency#F3#beta', 'Spectral#RelativePower#F3#gamma', 'Spectral#PeakFrequency#F4#delta', 'Spectral#Diff#F4#delta', 'Spectral#RelativePower#F4#theta', 'Spectral#Entropy#F4#theta', 'Spectral#Flatness#F4#theta', 'Spectral#Diff#F4#theta', 'Spectral#RelativePower#F4#alpha', 'Spectral#Entropy#F4#alpha', 'Spectral#EdgeFrequency#F4#alpha', 'Spectral#RelativePower#F4#beta', 'Spectral#Entropy#F4#beta', 'Spectral#Diff#F4#beta', 'Spectral#PeakFrequency#F4#gamma', 'Spectral#Entropy#F7#delta', 'Spectral#Diff#F7#delta', 'Spectral#Entropy#F7#theta', 'Spectral#Flatness#F7#theta', 'Spectral#Diff#F7#theta', 'Spectral#EdgeFrequency#F7#alpha', 'Spectral#Diff#F7#alpha', 'Spectral#Flatness#F7#beta', 'Spectral#PeakFrequency#F7#beta', 'Spectral#Diff#F7#beta', 'Spectral#RelativePower#F7#gamma', 'Spectral#Entropy#F7#gamma', 'Spectral#Flatness#F7#gamma', 'Spectral#EdgeFrequency#F7#gamma', 'Spectral#Diff#F7#gamma', 'Spectral#EdgeFrequency#F8#delta', 'Spectral#PeakFrequency#F8#delta', 'Spectral#Diff#F8#delta', 'Spectral#Entropy#F8#theta', 'Spectral#Flatness#F8#theta', 'Spectral#EdgeFrequency#F8#theta', 'Spectral#PeakFrequency#F8#theta', 'Spectral#Diff#F8#theta', 'Spectral#RelativePower#F8#alpha', 'Spectral#Entropy#F8#alpha', 'Spectral#Flatness#F8#alpha', 'Spectral#EdgeFrequency#F8#alpha', 'Spectral#PeakFrequency#F8#alpha', 'Spectral#Diff#F8#alpha', 'Spectral#Entropy#F8#beta']


out_path = './scheme3/cv'

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

    # Targets are float values; let's make with intervals of |26|
    min_target = np.min(targets)
    max_target = np.max(targets)
    target_classes_intervals = np.arange(int(min_target-1), int(max_target+1), 26)
    targets_classes = np.digitize(targets, target_classes_intervals)

    # Identify the minority class
    unique_targets_classes, counts = np.unique(targets_classes, return_counts=True)
    minority_class = unique_targets_classes[np.argmin(counts)]
    test_size = 1/n_splits
    n_test_per_target_class = int(np.round(test_size * counts[np.argmin(counts)]))

    # for each fold...
    for i in range(n_splits):
        # 1. Select 'n_test_per_target_class' per each target class to constitute the test set
        test_indices = []
        for target_class in unique_targets_classes:

            # Get the indices for this target class that have not been used for the test set yet
            target_indices = [i for i in indices if targets_classes[i] == target_class and i not in used_test_indices]

            if len(target_indices) < n_test_per_target_class:
                continue  # If the number of examples of this target class is less than 'n_test_per_target_class', do not select any; they will be used in the training set
            np.random.seed(random_state)
            Y = np.random.choice(target_indices, size=n_test_per_target_class, replace=False)
            test_indices.extend(Y)
        # Add the test indices for this fold to the list of used test indices
        used_test_indices.extend(test_indices)

        # Print targets distribution for test set
        print(f"Test set distribution:")
        test_dist = np.unique(targets_classes[test_indices], return_counts=True)
        print(test_dist)


        # 2. Identify the INSIGHT examples in the test set, and find remaining examples for training
        insight_indices = [i for i in test_indices if '_' in str(objects.index[i])]
        # Exclude other examples from the same subjects from the training set
        excluded_subjects = {objects.index[i].split('_')[0] for i in insight_indices if '_' in objects.index[i]}
        remaining_indices = [i for i in indices if objects.index[i].split('_')[0] not in excluded_subjects and i not in test_indices]

        # 3. Augment the remaining examples that can be selected for training
        #augmented_objects, augmented_targets = augment(objects.iloc[remaining_indices], targets.iloc[remaining_indices])
        #yield augmented_objects, objects.iloc[test_indices], augmented_targets, targets.iloc[test_indices]

        yield objects.iloc[remaining_indices], objects.iloc[test_indices], targets.iloc[remaining_indices], targets.iloc[test_indices]



def train_test_cv(model, objects, targets, folds: int, random_state:int):
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
        res.to_csv(join(fold_path, 'predictions_targets.csv'))

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
        plt.xlim(-41, 16)
        plt.ylim(-41, 16)
        plt.tight_layout()
        #plt.show()
        plt.savefig(join(fold_path, 'test.png'))

    # Print the average scores
    print(f'Average R2: {np.mean(r2_scores)} +/- {np.std(r2_scores)}')
    print(f'Average MSE: {np.mean(mse_scores)} +/- {np.std(mse_scores)}')
    print(f'Average MAE: {np.mean(mae_scores)} +/- {np.std(mae_scores)}')



# 1.1) Get all features
sapienza = read_all_features('Sapienza', multiples=True)
insight = read_all_features('INSIGHT', multiples=True)
features = pd.concat([sapienza, insight], axis=0)
features = features[FEATURES_SELECTED]
print("Read all features. Shape:", features.shape)

# 2) Get targets
insight_targets = read_brainage('INSIGHT')
insight_ages = read_ages('INSIGHT')
sapienza_targets = read_brainage('Sapienza')
sapienza_ages = read_ages('Sapienza')
targets = Series()
for index in features.index:
    if '$' in str(index):  # Multiples
        key = str(index).split('$')[0]  # remove the multiple
    else:  # Original
        key = index

    if '_' in str(key):  # insight
        key = int(key.split('_')[0])
        if key in insight_targets and key in insight_ages:
            targets.loc[index] = insight_targets[key] - insight_ages[key]
    elif 'PARTICIPANT' in str(key):  # sapienza
        if key in sapienza_targets and key in sapienza_ages:
            targets.loc[index] = sapienza_targets[key] - sapienza_ages[key]

# Drop subject_sessions with nans targets
targets = targets.dropna()
features = features.loc[targets.index]

"""
# Normalise feature vectors BEFORE
features = feature_wise_normalisation(features, method='min-max')
features = features.dropna(axis=1)


# 3) Data Augmentation in the underrepresented MMSE scores

# Histogram before
plt.hist(targets, bins=27, rwidth=0.8)
plt.title("Before")
plt.show()

#3. Data Augmentation method = SMOTE-R
features['target'] = targets  # Append column targets
features = features.reset_index(drop=True)  # make index sequential
features = features.dropna()
features = smogn.smoter(
    data = features,
    y = "target",
    k=5, 
    under_samp=False,
)
features = features.dropna()
targets = features['target'] # Drop column targets
features = features.drop(columns=['target'])
features = features.reset_index(drop=True)  # Drop index
targets = targets.reset_index(drop=True)  # Drop index)

# Histogram after
plt.hist(targets, bins=27, rwidth=0.8)
plt.title("After")
plt.show()

print("Features shape after DA:", features.shape)

# Normalise feature vectors AFTER
features = feature_wise_normalisation(features, method='min-max')
features = features.dropna(axis=1)
"""
features = features.dropna(axis=0)
features = features.dropna(axis=1)
targets = targets[features.index]
print("Features shape after dropping", features.shape)

# 5) Define model
model = GradientBoostingRegressor(n_estimators=300, max_depth=15, random_state=0, loss='absolute_error',
                                  learning_rate=0.04,)


# 5. Train and Test
train_test_cv(model, features, targets, folds=5, random_state=42)


