from pickle import dump

import numpy as np
from matplotlib import pyplot as plt
from pandas import Series
from sklearn.ensemble import GradientBoostingRegressor

from read import *
from read import read_all_features
from utils import feature_wise_normalisation

out_path = './scheme16'


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
    #plt.show()
    plt.savefig(join(out_path, 'train.png'))

    # 9. Serialize model
    with open(join(out_path, 'model.pkl'), 'wb') as f:
        dump(model, f)


# 1) Get all features
insight = read_all_features('INSIGHT')
brainlat = read_all_features('BrainLat')
miltiadous = read_all_features('Miltiadous Dataset')
sapienza = read_all_features('Sapienza')

# Multiples
#multiples = read_all_features_multiples()
# Perturb the multiple features, so they are not identical to the original ones
# These sigma values were defined based on similarity with the original features; the goal is to make them disimilar inasmuch as other examples from other subjects.
"""
jitter = lambda x: x + np.random.normal(0, 0.1, x.shape)
scaling = lambda x: x * np.random.normal(1, 0.04, x.shape)
print("Perturbing multiple examples...")
for feature in multiples.columns:
    data = multiples[feature].values
    data = jitter(data)
    data = scaling(data)
    multiples[feature] = data
"""
####

#features = pd.concat([insight, brainlat, miltiadous, multiples], axis=0)
#features = pd.concat([insight, brainlat, miltiadous], axis=0)
features = pd.concat([insight, brainlat, miltiadous, sapienza], axis=0)
print("Read all features. Shape:", features.shape)

# 1.1.) Select features
# FIXME
# kjpp + eldersly features selected (80)
#FEATURES_SELECTED = ['Spectral#RelativePower#F3#gamma', 'Hjorth#Complexity#T3', 'Spectral#PeakFrequency#O1#beta3', 'Spectral#Entropy#Pz#beta1', 'Spectral#RelativePower#Cz#beta2', 'Spectral#Diff#P4#beta2', 'Spectral#Flatness#T5#alpha2', 'Spectral#PeakFrequency#Fz#beta3', 'Spectral#EdgeFrequency#T3#delta', 'PLI#Temporal(L)-Occipital(L)#beta1', 'Spectral#RelativePower#C4#delta', 'Spectral#PeakFrequency#F8#alpha1', 'Spectral#EdgeFrequency#Pz#gamma', 'Spectral#PeakFrequency#Cz#gamma', 'Spectral#Flatness#T6#gamma', 'Spectral#RelativePower#Fz#delta', 'Spectral#EdgeFrequency#Fz#beta3', 'Spectral#EdgeFrequency#F8#beta3', 'Spectral#Diff#Cz#gamma', 'Hjorth#Activity#C3', 'Spectral#RelativePower#Cz#delta', 'Spectral#RelativePower#Fp2#gamma', 'Spectral#Entropy#F7#theta', 'PLI#Temporal(L)-Parietal(L)#alpha2', 'Spectral#RelativePower#T4#beta1', 'Spectral#RelativePower#Cz#gamma', 'Hjorth#Activity#P4', 'Spectral#RelativePower#Fz#gamma', 'Spectral#RelativePower#P3#theta', 'Spectral#EdgeFrequency#O2#beta2', 'Spectral#Diff#C4#beta1', 'Spectral#RelativePower#C3#gamma', 'Spectral#RelativePower#P4#beta3', 'Spectral#PeakFrequency#Fp2#beta2', 'Spectral#EdgeFrequency#T3#theta', 'Spectral#RelativePower#Fp1#beta1', 'Hjorth#Mobility#Pz', 'Spectral#RelativePower#Fpz#gamma', 'Spectral#Diff#T4#beta1', 'Spectral#Entropy#P3#alpha1', 'Spectral#Flatness#F4#beta2', 'Spectral#Entropy#F4#beta2', 'Spectral#RelativePower#C4#gamma', 'Spectral#RelativePower#Cz#beta3', 'Spectral#RelativePower#O1#alpha2', 'Spectral#PeakFrequency#Fz#gamma', 'Spectral#PeakFrequency#F4#delta', 'Spectral#RelativePower#P4#alpha1', 'Spectral#PeakFrequency#P3#gamma', 'Spectral#RelativePower#O2#beta3', 'Hjorth#Mobility#P4', 'Hjorth#Complexity#Fp2', 'Spectral#Diff#P3#beta1', 'Spectral#RelativePower#C3#beta1', 'Spectral#EdgeFrequency#Cz#beta3', 'Spectral#Diff#C3#theta', 'Spectral#RelativePower#Fp1#beta2', 'Spectral#EdgeFrequency#F4#delta', 'Spectral#RelativePower#P3#beta3', 'Spectral#RelativePower#F3#theta', 'Spectral#Entropy#O2#delta', 'Spectral#PeakFrequency#C4#beta1', 'Spectral#EdgeFrequency#Cz#gamma', 'Spectral#RelativePower#T6#beta1', 'Spectral#PeakFrequency#O2#theta', 'Spectral#Flatness#C4#gamma', 'Spectral#PeakFrequency#F7#gamma', 'Spectral#RelativePower#F7#gamma', 'Spectral#Diff#T5#beta1', 'Spectral#EdgeFrequency#Pz#beta3', 'Spectral#RelativePower#T3#beta3', 'Spectral#RelativePower#T4#gamma', 'Spectral#EdgeFrequency#F3#theta', 'Spectral#RelativePower#Fp2#beta1', 'Spectral#PeakFrequency#Fz#theta', 'Hjorth#Complexity#T5', 'Hjorth#Complexity#T4', 'Hjorth#Complexity#F8', 'Hjorth#Mobility#Cz', 'Spectral#PeakFrequency#Fp2#alpha1']

# 80 features from elders RFE
FEATURES_SELECTED = ['Spectral#RelativePower#C3#beta1', 'Spectral#EdgeFrequency#C3#beta3', 'Spectral#RelativePower#C3#gamma', 'Spectral#EdgeFrequency#C4#alpha1', 'Spectral#RelativePower#C4#beta3', 'Spectral#EdgeFrequency#C4#beta3', 'Spectral#EdgeFrequency#C4#gamma', 'Spectral#Flatness#Cz#theta', 'Spectral#PeakFrequency#Cz#theta', 'Spectral#EdgeFrequency#Cz#beta3', 'Spectral#EdgeFrequency#Cz#gamma', 'Spectral#PeakFrequency#Cz#gamma', 'Spectral#RelativePower#F3#beta1', 'Spectral#Diff#F4#delta', 'Spectral#RelativePower#F7#beta3', 'Spectral#EdgeFrequency#F7#beta3', 'Spectral#RelativePower#F7#gamma', 'Spectral#RelativePower#F8#beta1', 'Spectral#EdgeFrequency#F8#beta3', 'Spectral#RelativePower#Fp1#beta1', 'Spectral#EdgeFrequency#Fp1#beta3', 'Spectral#Diff#Fp2#delta', 'Spectral#RelativePower#Fp2#beta1', 'Spectral#RelativePower#Fp2#beta3', 'Spectral#Diff#Fpz#beta2', 'Spectral#Entropy#O1#delta', 'Spectral#RelativePower#O1#beta2', 'Spectral#EdgeFrequency#O1#beta2', 'Spectral#EdgeFrequency#O1#beta3', 'Spectral#RelativePower#O2#delta', 'Spectral#PeakFrequency#O2#alpha1', 'Spectral#RelativePower#O2#beta1', 'Spectral#RelativePower#O2#beta3', 'Spectral#Diff#P3#beta1', 'Spectral#RelativePower#P3#beta3', 'Spectral#RelativePower#Pz#alpha1', 'Spectral#EdgeFrequency#Pz#beta3', 'Spectral#RelativePower#T4#alpha1', 'Spectral#RelativePower#T4#beta3', 'Spectral#RelativePower#T4#gamma', 'Spectral#EdgeFrequency#T5#beta2', 'Hjorth#Complexity#T5', 'Hjorth#Complexity#P4', 'Hjorth#Complexity#F7', 'Hjorth#Complexity#T4', 'Hjorth#Complexity#F8', 'Hjorth#Complexity#T3', 'Hjorth#Mobility#P3', 'PLI#Frontal(L)-Temporal(R)#alpha1', 'PLI#Frontal(L)-Occipital(L)#alpha1', 'PLI#Frontal(R)-Temporal(R)#alpha1', 'PLI#Temporal(R)-Parietal(R)#alpha1', 'PLI#Temporal(R)-Occipital(L)#alpha1', 'PLI#Parietal(R)-Occipital(L)#alpha1', 'PLI#Occipital(L)-Occipital(R)#alpha1', 'PLI#Temporal(R)-Occipital(R)#alpha2', 'PLI#Parietal(R)-Occipital(L)#alpha2', 'COH#Frontal(L)-Frontal(R)#theta', 'COH#Frontal(L)-Occipital(L)#theta', 'COH#Frontal(L)-Occipital(R)#alpha1', 'COH#Frontal(R)-Occipital(L)#alpha1', 'COH#Parietal(R)-Occipital(L)#alpha1', 'COH#Frontal(L)-Frontal(R)#alpha2', 'COH#Frontal(L)-Occipital(R)#alpha2', 'COH#Parietal(R)-Occipital(L)#alpha2', 'COH#Parietal(R)-Occipital(R)#alpha2', 'COH#Occipital(L)-Occipital(R)#alpha2', 'COH#Frontal(L)-Occipital(L)#beta1', 'COH#Temporal(R)-Parietal(R)#beta1', 'COH#Parietal(R)-Occipital(R)#beta1', 'COH#Frontal(L)-Parietal(L)#beta2', 'COH#Frontal(R)-Occipital(L)#beta2', 'COH#Frontal(L)-Temporal(R)#beta3', 'COH#Frontal(L)-Parietal(L)#beta3', 'COH#Frontal(L)-Occipital(L)#beta3', 'COH#Frontal(L)-Occipital(R)#beta3', 'COH#Frontal(R)-Occipital(L)#beta3', 'COH#Temporal(L)-Occipital(R)#beta3', 'COH#Frontal(L)-Occipital(R)#gamma', 'COH#Frontal(R)-Occipital(R)#gamma']

# 80 features from children RFE
#FEATURES_SELECTED += ['Spectral#EdgeFrequency#C4#theta', 'Spectral#PeakFrequency#C4#gamma', 'Spectral#RelativePower#Cz#gamma', 'Spectral#RelativePower#F3#delta', 'Spectral#RelativePower#F3#alpha1', 'Spectral#PeakFrequency#F7#beta3', 'Spectral#EdgeFrequency#F8#gamma', 'Spectral#RelativePower#Fp1#delta', 'Spectral#RelativePower#Fp1#alpha1', 'Spectral#Entropy#Fp2#beta3', 'Spectral#RelativePower#Fz#alpha1', 'Spectral#PeakFrequency#O1#alpha1', 'Spectral#EdgeFrequency#O1#beta3', 'Spectral#RelativePower#O2#theta', 'Spectral#Entropy#O2#beta2', 'Spectral#Flatness#O2#beta2', 'Spectral#EdgeFrequency#O2#beta3', 'Spectral#PeakFrequency#O2#gamma', 'Spectral#EdgeFrequency#P3#gamma', 'Spectral#PeakFrequency#P4#alpha1', 'Spectral#EdgeFrequency#P4#gamma', 'Spectral#PeakFrequency#Pz#alpha1', 'Spectral#EdgeFrequency#T4#beta2', 'Spectral#RelativePower#T5#beta2', 'Spectral#Flatness#T5#beta3', 'Hjorth#Activity#P4', 'Hjorth#Activity#F4', 'Hjorth#Activity#C4', 'Hjorth#Activity#F8', 'Hjorth#Activity#C3', 'Hjorth#Mobility#O2', 'Hjorth#Mobility#T3', 'Hjorth#Mobility#Fz', 'Hjorth#Mobility#Cz', 'Hjorth#Mobility#T6', 'Hjorth#Mobility#Fp1', 'Hjorth#Mobility#C4', 'Hjorth#Mobility#P3', 'Hjorth#Mobility#F8', 'Hjorth#Mobility#O1', 'Hjorth#Mobility#T4', 'Hjorth#Complexity#O2', 'Hjorth#Complexity#F3', 'Hjorth#Complexity#Fz', 'PLI#Temporal(L)-Occipital(R)#alpha1', 'PLI#Parietal(L)-Occipital(L)#alpha1', 'COH#Frontal(L)-Frontal(R)#delta', 'COH#Frontal(L)-Parietal(R)#delta', 'COH#Temporal(L)-Parietal(R)#delta', 'COH#Temporal(R)-Parietal(L)#delta', 'COH#Temporal(R)-Parietal(R)#delta', 'COH#Temporal(L)-Occipital(L)#theta', 'COH#Temporal(R)-Parietal(R)#theta', 'COH#Temporal(R)-Occipital(L)#theta', 'COH#Temporal(R)-Occipital(R)#theta', 'COH#Frontal(L)-Temporal(R)#alpha1', 'COH#Frontal(L)-Parietal(R)#alpha1', 'COH#Frontal(L)-Occipital(L)#alpha1', 'COH#Frontal(R)-Temporal(L)#alpha1', 'COH#Frontal(R)-Parietal(L)#alpha1', 'COH#Temporal(L)-Parietal(R)#alpha1', 'COH#Frontal(L)-Parietal(R)#alpha2', 'COH#Frontal(L)-Occipital(L)#alpha2', 'COH#Frontal(R)-Parietal(L)#alpha2', 'COH#Frontal(R)-Occipital(R)#alpha2', 'COH#Frontal(L)-Occipital(L)#beta1', 'COH#Frontal(R)-Parietal(L)#beta1', 'COH#Frontal(R)-Occipital(R)#beta1', 'COH#Temporal(L)-Parietal(L)#beta1', 'COH#Temporal(R)-Parietal(L)#beta1', 'COH#Temporal(R)-Parietal(R)#beta1', 'COH#Frontal(L)-Frontal(R)#beta2', 'COH#Frontal(L)-Temporal(L)#beta2', 'COH#Frontal(L)-Parietal(R)#beta2', 'COH#Frontal(L)-Occipital(L)#beta2', 'COH#Frontal(R)-Occipital(R)#beta2', 'COH#Frontal(L)-Parietal(R)#beta3', 'COH#Frontal(L)-Occipital(L)#beta3', 'COH#Frontal(R)-Parietal(L)#beta3', 'COH#Frontal(R)-Occipital(L)#gamma']
#FEATURES_SELECTED = set(FEATURES_SELECTED)
#FEATURES_SELECTED = list(FEATURES_SELECTED)

features = features[FEATURES_SELECTED]
print("Number of features selected:", len(features.columns))
# Drop NaN values
features = features.dropna(axis=0)

# Save stochastic pattern
stochastics = DataFrame([features.mean(), features.std(), features.min(), features.max()], index=['mean', 'std', 'min', 'max'])
#stochastics.to_csv('elderly_stochastic_pattern.csv')

# 2) Get targets
insight_targets = read_mmse('INSIGHT')
brainlat_targets = read_mmse('BrainLat')
miltiadous_targets = read_mmse('Miltiadous Dataset')
sapienza_targets = read_mmse('Sapienza')
targets = Series()
for index in features.index:
    if '_' in str(index):  # insight
        key = int(index.split('_')[0])
        if key in insight_targets:
            targets.loc[index] = insight_targets[key]
    elif '-' in str(index):  # brainlat
        if index in brainlat_targets:
            targets.loc[index] = brainlat_targets[index]
    elif 'PARTICIPANT' in str(index):  # sapienza
        if index in sapienza_targets:
            targets.loc[index] = sapienza_targets[index]
    else:  # miltiadous
        # parse e.g. 24 -> 'sub-024'; 1 -> 'sub-001'
        if '$' in str(index):  # EXTRA: multiple examples, remove the $ and the number after it; the target is the same
            key = 'sub-' + str(str(index).split('$')[0]).zfill(3)
        else:
            key = 'sub-' + str(index).zfill(3)
        if key:
            targets.loc[index] = miltiadous_targets[key]

# Drop subject_sessions with nans targets
targets = targets.dropna()
features = features.loc[targets.index]

# Normalise feature vectors BEFORE
features = feature_wise_normalisation(features, method='min-max')
features = features.dropna(axis=1)


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

# Save perfect score stochastic pattern (mmse == 30)
mmse30_features = [x[0] for x in dataset if x[1] == 30]
mmse30_features = np.array(mmse30_features)
mmse30_stochastics = DataFrame([mmse30_features.mean(axis=0),
                                mmse30_features.std(axis=0),
                                mmse30_features.min(axis=0),
                                mmse30_features.max(axis=0)],
                               index=['mean', 'std', 'min', 'max'],
                               columns=feature_names)
#mmse30_stochastics.to_csv('elderly_mmse30_stochastic_pattern.csv')


# 3) Define model
model = GradientBoostingRegressor(n_estimators=200, max_depth=10, random_state=0, loss='absolute_error',
                                  learning_rate=0.04,)

# 4) Train a model and save
train_full_dataset(model, dataset)


