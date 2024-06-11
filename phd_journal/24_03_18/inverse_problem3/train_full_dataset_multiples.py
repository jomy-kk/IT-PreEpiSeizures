from pickle import dump

import numpy as np
from matplotlib import pyplot as plt
from pandas import Series
from sklearn.ensemble import GradientBoostingRegressor

from read import *
from read import read_all_features
from utils import feature_wise_normalisation

out_path = './scheme35'


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
miltiadous = read_all_features('Miltiadous Dataset', multiples=True)
brainlat = read_all_features('BrainLat', multiples=True)
sapienza = read_all_features('Sapienza', multiples=True)
insight = read_all_features('INSIGHT', multiples=True)
features = pd.concat([brainlat, miltiadous, sapienza, insight], axis=0)
print("Read all features. Shape:", features.shape)


"""
# Features transformation
# 80 features from elders RFE
FEATURES_SELECTED_OLD = ['Spectral#RelativePower#C3#beta1', 'Spectral#EdgeFrequency#C3#beta3', 'Spectral#RelativePower#C3#gamma', 'Spectral#EdgeFrequency#C4#alpha1', 'Spectral#RelativePower#C4#beta3', 'Spectral#EdgeFrequency#C4#beta3', 'Spectral#EdgeFrequency#C4#gamma', 'Spectral#Flatness#Cz#theta', 'Spectral#PeakFrequency#Cz#theta', 'Spectral#EdgeFrequency#Cz#beta3', 'Spectral#EdgeFrequency#Cz#gamma', 'Spectral#PeakFrequency#Cz#gamma', 'Spectral#RelativePower#F3#beta1', 'Spectral#Diff#F4#delta', 'Spectral#RelativePower#F7#beta3', 'Spectral#EdgeFrequency#F7#beta3', 'Spectral#RelativePower#F7#gamma', 'Spectral#RelativePower#F8#beta1', 'Spectral#EdgeFrequency#F8#beta3', 'Spectral#RelativePower#Fp1#beta1', 'Spectral#EdgeFrequency#Fp1#beta3', 'Spectral#Diff#Fp2#delta', 'Spectral#RelativePower#Fp2#beta1', 'Spectral#RelativePower#Fp2#beta3', 'Spectral#Diff#Fpz#beta2', 'Spectral#Entropy#O1#delta', 'Spectral#RelativePower#O1#beta2', 'Spectral#EdgeFrequency#O1#beta2', 'Spectral#EdgeFrequency#O1#beta3', 'Spectral#RelativePower#O2#delta', 'Spectral#PeakFrequency#O2#alpha1', 'Spectral#RelativePower#O2#beta1', 'Spectral#RelativePower#O2#beta3', 'Spectral#Diff#P3#beta1', 'Spectral#RelativePower#P3#beta3', 'Spectral#RelativePower#Pz#alpha1', 'Spectral#EdgeFrequency#Pz#beta3', 'Spectral#RelativePower#T4#alpha1', 'Spectral#RelativePower#T4#beta3', 'Spectral#RelativePower#T4#gamma', 'Spectral#EdgeFrequency#T5#beta2', 'Hjorth#Complexity#T5', 'Hjorth#Complexity#P4', 'Hjorth#Complexity#F7', 'Hjorth#Complexity#T4', 'Hjorth#Complexity#F8', 'Hjorth#Complexity#T3', 'Hjorth#Mobility#P3', 'PLI#Frontal(L)-Temporal(R)#alpha1', 'PLI#Frontal(L)-Occipital(L)#alpha1', 'PLI#Frontal(R)-Temporal(R)#alpha1', 'PLI#Temporal(R)-Parietal(R)#alpha1', 'PLI#Temporal(R)-Occipital(L)#alpha1', 'PLI#Parietal(R)-Occipital(L)#alpha1', 'PLI#Occipital(L)-Occipital(R)#alpha1', 'PLI#Temporal(R)-Occipital(R)#alpha2', 'PLI#Parietal(R)-Occipital(L)#alpha2', 'COH#Frontal(L)-Frontal(R)#theta', 'COH#Frontal(L)-Occipital(L)#theta', 'COH#Frontal(L)-Occipital(R)#alpha1', 'COH#Frontal(R)-Occipital(L)#alpha1', 'COH#Parietal(R)-Occipital(L)#alpha1', 'COH#Frontal(L)-Frontal(R)#alpha2', 'COH#Frontal(L)-Occipital(R)#alpha2', 'COH#Parietal(R)-Occipital(L)#alpha2', 'COH#Parietal(R)-Occipital(R)#alpha2', 'COH#Occipital(L)-Occipital(R)#alpha2', 'COH#Frontal(L)-Occipital(L)#beta1', 'COH#Temporal(R)-Parietal(R)#beta1', 'COH#Parietal(R)-Occipital(R)#beta1', 'COH#Frontal(L)-Parietal(L)#beta2', 'COH#Frontal(R)-Occipital(L)#beta2', 'COH#Frontal(L)-Temporal(R)#beta3', 'COH#Frontal(L)-Parietal(L)#beta3', 'COH#Frontal(L)-Occipital(L)#beta3', 'COH#Frontal(L)-Occipital(R)#beta3', 'COH#Frontal(R)-Occipital(L)#beta3', 'COH#Temporal(L)-Occipital(R)#beta3', 'COH#Frontal(L)-Occipital(R)#gamma', 'COH#Frontal(R)-Occipital(R)#gamma']
FEATURES_SELECTED = []
for feature in FEATURES_SELECTED_OLD:
    if 'alpha1' in feature or 'alpha2' in feature or 'beta1' in feature or 'beta2' in feature or 'beta3' in feature:
        feature = feature[:-1]
    FEATURES_SELECTED.append(feature)
FEATURES_SELECTED = list(set(FEATURES_SELECTED))
features = features[FEATURES_SELECTED]
print("Number of features selected:", len(features.columns))
features = features.dropna(axis=0)
"""

# new: MI + RFE (932 -> 200 -> 80 features)
FEATURES_SELECTED = afeatures = features[FEATURES_SELECTED]
features = features.dropna(axis=0)

# 2) Get targets
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

# Drop subject_sessions with nans targets
targets = targets.dropna()
features = features.loc[targets.index]

# Normalise feature vectors BEFORE
features = feature_wise_normalisation(features, method='min-max')
features = features.dropna(axis=1)

"""
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
"""
######################
# SMOTE-C

from imblearn.over_sampling import SMOTE
# let's make targe 15->12
#targets = targets.replace(15, 12)
smote = SMOTE(random_state=42, k_neighbors=5, sampling_strategy='auto')

# Histogram before
plt.hist(targets, bins=30)
plt.title("Before")
plt.ylim((0, 230))
plt.show()

features, targets = smote.fit_resample(features, targets)

# Histogram after
plt.hist(targets, bins=30)
plt.title("After")
plt.ylim((0, 230))
plt.show()

######################

######################
# SMOTE-R
"""
import smogn

# Histogram before
plt.hist(targets, bins=30)
plt.title("Before")
plt.ylim((0, 230))
plt.show()

# Append column targets
features['target'] = targets
# make index sequential
features = features.reset_index(drop=True)
features = features.dropna()
# Apply SMOGN to balance the datas
features = smogn.smoter(
    data = features,
    k=1,
    samp_method = "extreme",
    y = 'target'
)

# Drop nans
features = features.dropna(axis=0)

# Drop column targets
targets = features['target']
features = features.drop(columns=['target'])

# Drop index
features = features.reset_index(drop=True)
targets = targets.reset_index(drop=True)

# Histogram after
plt.hist(targets, bins=30)
plt.title("After")
plt.ylim((0, 230))
plt.show()
"""
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
features = features.to_numpy(copy=True)
dataset = []
for i, session in enumerate(sessions):
    dataset.append((features[i], targets[session]))

# Save perfect score stochastic pattern (mmse == 30)
"""
mmse30_features = [x[0] for x in dataset if x[1] == 30]
mmse30_features = np.array(mmse30_features)
mmse30_stochastics = DataFrame([mmse30_features.mean(axis=0),
                                mmse30_features.std(axis=0),
                                mmse30_features.min(axis=0),
                                mmse30_features.max(axis=0)],
                               index=['mean', 'std', 'min', 'max'],
                               columns=feature_names)
mmse30_stochastics.to_csv('elderly_mmse30_stochastic_pattern.csv')
"""

# 3) Define model
model = GradientBoostingRegressor(n_estimators=300, max_depth=15, random_state=0, loss='absolute_error',
                                  learning_rate=0.04,)

# 4) Train a model and save
train_full_dataset(model, dataset)


