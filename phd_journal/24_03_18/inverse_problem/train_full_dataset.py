from pickle import dump

import numpy as np
from matplotlib import pyplot as plt
from pandas import Series
from sklearn.ensemble import GradientBoostingRegressor

from read import *
from read import read_all_features
from utils import feature_wise_normalisation


def train_full_dataset(model, dataset):
    print(model)

    # Separate features and targets
    features = np.array([x[0] for x in dataset])
    targets = np.array([x[1] for x in dataset])
    print("Features shape:", features.shape)
    print("Targets shape:", targets.shape)

    # 5. Train the model only with the selected features
    model.fit(features, targets)

    # 6. Test the model
    predictions = model.predict(features)
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    # 6.1) Mean Squared Error (MSE)
    mse = mean_squared_error(targets, predictions)
    # 6.2) Mean Absolute Error (MAE)
    mae = mean_absolute_error(targets, predictions)
    # 6.3) R2 Score
    r2 = r2_score(targets, predictions)

    # 7. Print results
    print(f'Train MSE: {mse}')
    print(f'Train MAE: {mae}')
    print(f'Train R2: {r2}')
    print('---------------------------------')

    # 8. Plot regression between ground truth and predictions with seaborn and draw regression curve
    import seaborn as sns
    plt.figure(figsize=((5.2,5)))
    sns.regplot(x=targets, y=predictions, scatter_kws={'alpha': 0.3})
    #plt.title(str(model))
    plt.xlabel('True MMSE (units)')
    plt.ylabel('Predicted MMSE (units)')
    plt.xlim(0, 30)
    plt.ylim(0, 30)
    plt.tight_layout()
    plt.show()

    # 9. Serialize model
    with open('NEW_model_80feat.pkl', 'wb') as f:
        dump(model, f)


# 1) Get all features
insight = read_all_features('INSIGHT')
brainlat = read_all_features('BrainLat')
miltiadous = read_all_features('Miltiadous Dataset')
features = pd.concat([insight, brainlat, miltiadous], axis=0)
print("Read all features. Shape:", features.shape)

# 1.1.) Select features
# FIXME
#FEATURES_SELECTED = ['Spectral#RelativePower#C3#beta1', 'Spectral#Diff#C4#beta1', 'Spectral#RelativePower#C4#gamma', 'Spectral#RelativePower#Cz#delta', 'Spectral#EdgeFrequency#Cz#beta3', 'Spectral#EdgeFrequency#Cz#gamma', 'Spectral#PeakFrequency#Cz#gamma', 'Spectral#RelativePower#F3#theta', 'Spectral#EdgeFrequency#F3#theta', 'Spectral#RelativePower#F3#gamma', 'Spectral#EdgeFrequency#F4#delta', 'Spectral#PeakFrequency#F4#delta', 'Spectral#RelativePower#F7#gamma', 'Spectral#PeakFrequency#F7#gamma', 'Spectral#PeakFrequency#F8#alpha1', 'Spectral#EdgeFrequency#F8#beta3', 'Spectral#RelativePower#Fp1#beta1', 'Spectral#RelativePower#Fp1#beta2', 'Spectral#RelativePower#Fp2#beta1', 'Spectral#RelativePower#Fp2#gamma', 'Spectral#RelativePower#Fpz#gamma', 'Spectral#EdgeFrequency#Fz#beta3', 'Spectral#PeakFrequency#Fz#beta3', 'Spectral#RelativePower#Fz#gamma', 'Spectral#RelativePower#O1#alpha2', 'Spectral#EdgeFrequency#O2#beta2', 'Spectral#RelativePower#O2#beta3', 'Spectral#RelativePower#P3#theta', 'Spectral#Entropy#P3#alpha1', 'Spectral#Diff#P3#beta1', 'Spectral#RelativePower#P3#beta3', 'Spectral#RelativePower#P4#alpha1', 'Spectral#RelativePower#P4#beta3', 'Spectral#Entropy#Pz#beta1', 'Spectral#EdgeFrequency#Pz#beta3', 'Spectral#EdgeFrequency#T3#theta', 'Spectral#RelativePower#T3#beta3', 'Spectral#RelativePower#T4#beta1', 'Spectral#Diff#T4#beta1', 'Spectral#RelativePower#T4#gamma', 'Spectral#Diff#T5#beta1', 'Spectral#RelativePower#T6#beta1', 'Spectral#Flatness#T6#gamma', 'Hjorth#Complexity#T5', 'Hjorth#Complexity#Fp2', 'Hjorth#Complexity#T4', 'Hjorth#Complexity#F8', 'Hjorth#Complexity#T3', 'Hjorth#Mobility#P4', 'Hjorth#Mobility#Pz']
# kjpp + eldersly features selected (80)
FEATURES_SELECTED = ['Spectral#RelativePower#F3#gamma', 'Hjorth#Complexity#T3', 'Spectral#PeakFrequency#O1#beta3', 'Spectral#Entropy#Pz#beta1', 'Spectral#RelativePower#Cz#beta2', 'Spectral#Diff#P4#beta2', 'Spectral#Flatness#T5#alpha2', 'Spectral#PeakFrequency#Fz#beta3', 'Spectral#EdgeFrequency#T3#delta', 'PLI#Temporal(L)-Occipital(L)#beta1', 'Spectral#RelativePower#C4#delta', 'Spectral#PeakFrequency#F8#alpha1', 'Spectral#EdgeFrequency#Pz#gamma', 'Spectral#PeakFrequency#Cz#gamma', 'Spectral#Flatness#T6#gamma', 'Spectral#RelativePower#Fz#delta', 'Spectral#EdgeFrequency#Fz#beta3', 'Spectral#EdgeFrequency#F8#beta3', 'Spectral#Diff#Cz#gamma', 'Hjorth#Activity#C3', 'Spectral#RelativePower#Cz#delta', 'Spectral#RelativePower#Fp2#gamma', 'Spectral#Entropy#F7#theta', 'PLI#Temporal(L)-Parietal(L)#alpha2', 'Spectral#RelativePower#T4#beta1', 'Spectral#RelativePower#Cz#gamma', 'Hjorth#Activity#P4', 'Spectral#RelativePower#Fz#gamma', 'Spectral#RelativePower#P3#theta', 'Spectral#EdgeFrequency#O2#beta2', 'Spectral#Diff#C4#beta1', 'Spectral#RelativePower#C3#gamma', 'Spectral#RelativePower#P4#beta3', 'Spectral#PeakFrequency#Fp2#beta2', 'Spectral#EdgeFrequency#T3#theta', 'Spectral#RelativePower#Fp1#beta1', 'Hjorth#Mobility#Pz', 'Spectral#RelativePower#Fpz#gamma', 'Spectral#Diff#T4#beta1', 'Spectral#Entropy#P3#alpha1', 'Spectral#Flatness#F4#beta2', 'Spectral#Entropy#F4#beta2', 'Spectral#RelativePower#C4#gamma', 'Spectral#RelativePower#Cz#beta3', 'Spectral#RelativePower#O1#alpha2', 'Spectral#PeakFrequency#Fz#gamma', 'Spectral#PeakFrequency#F4#delta', 'Spectral#RelativePower#P4#alpha1', 'Spectral#PeakFrequency#P3#gamma', 'Spectral#RelativePower#O2#beta3', 'Hjorth#Mobility#P4', 'Hjorth#Complexity#Fp2', 'Spectral#Diff#P3#beta1', 'Spectral#RelativePower#C3#beta1', 'Spectral#EdgeFrequency#Cz#beta3', 'Spectral#Diff#C3#theta', 'Spectral#RelativePower#Fp1#beta2', 'Spectral#EdgeFrequency#F4#delta', 'Spectral#RelativePower#P3#beta3', 'Spectral#RelativePower#F3#theta', 'Spectral#Entropy#O2#delta', 'Spectral#PeakFrequency#C4#beta1', 'Spectral#EdgeFrequency#Cz#gamma', 'Spectral#RelativePower#T6#beta1', 'Spectral#PeakFrequency#O2#theta', 'Spectral#Flatness#C4#gamma', 'Spectral#PeakFrequency#F7#gamma', 'Spectral#RelativePower#F7#gamma', 'Spectral#Diff#T5#beta1', 'Spectral#EdgeFrequency#Pz#beta3', 'Spectral#RelativePower#T3#beta3', 'Spectral#RelativePower#T4#gamma', 'Spectral#EdgeFrequency#F3#theta', 'Spectral#RelativePower#Fp2#beta1', 'Spectral#PeakFrequency#Fz#theta', 'Hjorth#Complexity#T5', 'Hjorth#Complexity#T4', 'Hjorth#Complexity#F8', 'Hjorth#Mobility#Cz', 'Spectral#PeakFrequency#Fp2#alpha1']
features = features[FEATURES_SELECTED]
print("Number of features selected:", len(features.columns))
# Drop NaN values
features = features.dropna(axis=0)

# Normalise feature vectors
features = feature_wise_normalisation(features, method='min-max')
features = features.dropna(axis=1)

# Save stochastic pattern
stochastics = DataFrame([features.mean(), features.std(), features.min(), features.max()], index=['mean', 'std', 'min', 'max'])
stochastics.to_csv('elderly_stochastic_pattern.csv')

# 2) Get targets
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
        key = 'sub-' + str(index).zfill(3)
        if key:
            targets.loc[index] = miltiadous_targets[key]

# Drop subject_sessions with nans targets
targets = targets.dropna()
features = features.loc[targets.index]

# 3) Data Augmentation in the underrepresented MMSE groups
# MMSE groups: 0-9, 9-15, 15-20, 20-24, 24-26, 26-30
# We'll augment the 0-9, 9-15, 15-20, 20-24, 24-26 groups because they are underrepresented
# We'll augment them until they have the same number of samples as the 26-30 group

# Get the number of samples in each group
mmse_groups = ((0, 9), (10, 15), (16, 20), (21, 25), (26, 30))
mmse_groups_samples = [len(targets[(mmse[0] <= targets) & (targets <= mmse[1])]) for mmse in mmse_groups]
max_samples = max(mmse_groups_samples)

# Augment the underrepresented groups
for i, mmse_group in enumerate(mmse_groups):
    if mmse_groups_samples[i] < max_samples:
        # Get the number of samples to augment
        n_samples_to_augment = max_samples - mmse_groups_samples[i]
        # Get the samples to augment
        samples = features[targets.between(mmse_group[0], mmse_group[1])]
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

# Assert that the number of samples in each group is the same
print("Number of samples before augmentation:")
print(mmse_groups)
print(mmse_groups_samples)
mmse_groups_samples_after = [len(targets[(mmse[0] <= targets) & (targets <= mmse[1])]) for mmse in mmse_groups]
assert all([samples == max_samples for samples in mmse_groups_samples_after])
print("Number of samples after augmentation:")
print(mmse_groups)
print(mmse_groups_samples_after)

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

# Save perfect score stochastic pattern (mmse == 30)
mmse30_features = [x[0] for x in dataset if x[1] == 30]
mmse30_features = np.array(mmse30_features)
mmse30_stochastics = DataFrame([mmse30_features.mean(axis=0),
                                mmse30_features.std(axis=0),
                                mmse30_features.min(axis=0),
                                mmse30_features.max(axis=0)],
                               index=['mean', 'std', 'min', 'max'],
                               columns=feature_names)
mmse30_stochastics.to_csv('elderly_mmse30_stochastic_pattern.csv')


# 3) Define model
model = GradientBoostingRegressor(n_estimators=200, max_depth=10, random_state=0, loss='absolute_error',
                                  learning_rate=0.04,)

# 4) Train a model and save
train_full_dataset(model, dataset)


