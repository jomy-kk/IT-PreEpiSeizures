from os import mkdir
from os.path import exists
from pickle import dump

import seaborn as sns
from imblearn.over_sampling import SMOTE
from matplotlib import pyplot as plt
from pandas import Series
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from read import *
from read import read_all_features
from utils import feature_wise_normalisation


out_path = './scheme57/cv'

PROCESS_NUMBER = 8  # FOR MULTIPROCESSING
N_CORES = 8  # FOR MULTIPROCESSING

def augment(features, targets):
    # 4) Data Augmentation in the underrepresented MMSE scores

    # Histogram before
    #plt.hist(targets, bins=27, rwidth=0.8)
    #plt.title("Before")
    #plt.show()

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
            #print(f"Interpolated {len(new_features)} examples for target {new_target}, from targets {lower_target} and {upper_target}")

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
    #plt.hist(targets, bins=27, rwidth=0.8)
    #plt.title("After interpolation of missing targets")
    #plt.show()

    # 4.2. Data Augmentation method = SMOTE-C
    for k in (5, 4, 3, 2, 1):
        try:
            smote = SMOTE(random_state=42, k_neighbors=k, sampling_strategy='auto')
            features, targets = smote.fit_resample(features, targets)
            print(f"SMOTE Worked with k={k}")
            break
        except ValueError as e:
            pass
            #print(f"Did not work with k={k}")

    # Histogram after
    #plt.hist(targets, bins=27, rwidth=0.8)
    #plt.title("After")
    #plt.show()

    # Normalisation after DA
    features = feature_wise_normalisation(features, method='min-max')
    features = features.dropna(axis=1)

    return features, targets


def custom_loocv(objects, targets, start, end, random_state=42):
    """
    Leave one out Cross-Validation with:
    a) ensures that the same subject is not present in both training and test sets.
    b) Data Augmentation on-the-fly on training sets.

    Args:
        objects: A DataFrame of feature vectors
        targets: A Series of target values
        start: The starting index of the objects to be considered for test sets
        end: The ending index of the objects to be considered for test sets

    Returns:
        The augmented training objects, test objects, training targets, and test targets.
    """

    # for each fold...
    for i in range(start, end):

        # 1) Select the i for the test set
        test_indices = [i, ]
        test_subject_code, test_multiple = objects.index[i].split('$')
        if '_' in test_subject_code:  # if INSIGHT
            test_subject_code = test_subject_code.split('_')[0]
        objects_test, targets_test = objects.iloc[test_indices], targets.iloc[test_indices]


        # 2) Select all others for the training set
        train_indices = [j for j in range(len(targets)) if j != i]

        # a) Ensure that the same subject is not present in both training and test sets
        # In INSIGHT, subject codes are identifiable from what's left of '_' in the index
        # In other datasets, independence check has to be done.
        for j in train_indices:
            subject_code, multiple = objects.index[j].split('$')
            if '_' in subject_code:  # if INSIGHT
                subject_code = subject_code.split('_')[0]
                if subject_code == test_subject_code:
                    train_indices.remove(j)
            else:
                if subject_code == test_subject_code:
                    if '-' in str(subject_code):  # brainlat
                        independents = brainlat_independents.loc[subject_code]
                    elif 'PARTICIPANT' in str(subject_code):  # sapienza
                        independents = sapienza_independents.loc[subject_code]
                    else:  # miltiadous
                        independents = miltiadous_independents.loc[subject_code]
                    if (multiple, test_multiple) not in independents and (test_multiple, multiple) not in independents:
                        train_indices.remove(j)
                else:
                    pass

        objects_train, targets_train = objects.iloc[train_indices], targets.iloc[train_indices]

        # b) Augment the training examples
        print("Train examples before augmentation:", len(objects_train))
        objects_train, targets_train = augment(objects_train, targets_train)
        print("Train examples after augmentation:", len(objects_train))

        yield objects_train, objects_test, targets_train, targets_test


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
# 1.2. Which multiples = all (check for independence is done during CV)
# 1.3. Which features = FEATURES_SELECTED
miltiadous = read_all_features('Miltiadous Dataset', multiples=True)
brainlat = read_all_features('BrainLat', multiples=True)
sapienza = read_all_features('Sapienza', multiples=True)
insight = read_all_features('INSIGHT', multiples=True)
features = pd.concat([brainlat, miltiadous, sapienza, insight], axis=0)
features = features[FEATURES_SELECTED]
features = features.dropna(axis=0)
print("Features Shape:", features.shape)

# 1.1.) Load independence tables
miltiadous_independents = pd.read_csv(join(common_datasets_path, 'Miltiadous Dataset', 'features', 'new_safe_multiples.csv'), index_col=0)
miltiadous_independents.index = [format(n, '03') for n in miltiadous_independents.index]
brainlat_independents = pd.read_csv(join(common_datasets_path, 'BrainLat', 'features', 'new_safe_multiples.csv'), index_col=0)
sapienza_independents = pd.read_csv(join(common_datasets_path, 'Sapienza', 'features', 'new_safe_multiples.csv'), index_col=0)


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
objects = features

# 4) Define model
model = GradientBoostingRegressor(n_estimators=300, max_depth=15, random_state=0, loss='absolute_error',
                                  learning_rate=0.04, )


# EXTRA: Reduce time complexity
# From examples with target classes higher than 200, randomly choose only 200
for mmse in range(31):
    if len(targets[targets == mmse]) > 200:
        objects_this_mmse = objects[targets == mmse]
        objects_this_mmse = objects_this_mmse.sample(n=200, random_state=42)
        objects = pd.concat([objects[targets != mmse], objects_this_mmse])
targets = targets.loc[objects.index]

print("Features Shape after reducing time complexity:", objects.shape)


# 5) Cross-Validation
N_FOLDS = int(len(objects) / N_CORES)
start = int(N_FOLDS * (PROCESS_NUMBER - 1))
end = int(N_FOLDS * PROCESS_NUMBER)

print("PROCESS_NUMBER:", PROCESS_NUMBER)
print("Test sets will go from", start, "to", end)

predictions = []
for i, (train_objects, test_objects, train_targets, test_targets) in (
        enumerate(custom_loocv(objects, targets, start, end, random_state=42))):
    print(f"Fold {i+1}/{N_FOLDS}")

    # Train the model
    print(f"Train examples: {len(train_objects)}")
    model.fit(train_objects, train_targets)

    # Test the model
    print(f"Test examples: {len(test_objects)}")
    prediction = model.predict(test_objects)
    predictions.extend(prediction)

    # Print absolute error and squared error
    print("Absolute Error:", mean_absolute_error(test_targets, prediction))
    print("Squared Error:", mean_squared_error(test_targets, prediction))

test_targets = targets[start:end]
# save Dataframe predictions | targets of test set
res = pd.DataFrame({'predictions': predictions, 'targets': test_targets})
res.to_csv(join(out_path, f'predictions_targets_{PROCESS_NUMBER}.csv'))

"""
# Print the average scores
r2 = r2_score(targets, predictions)  # R2
print(f'Average R2: {r2}')
mse = mean_squared_error(targets, predictions)  # MSE
print(f'Average MSE: {mse}')
mae = mean_absolute_error(targets, predictions)  # MAE
print(f'Average MAE: {mae}')

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
"""