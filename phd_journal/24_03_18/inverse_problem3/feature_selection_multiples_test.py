import numpy as np
from imblearn.over_sampling import SMOTE
from matplotlib import pyplot as plt
from pandas import Series
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.feature_selection import RFE, VarianceThreshold, SelectKBest, SelectPercentile, r_regression, f_regression, \
    mutual_info_regression
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split, cross_val_score, KFold, ShuffleSplit

from read import *
from read import read_all_features
from utils import feature_wise_normalisation

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


# Feature Selection
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
features = features[FEATURES_SELECTED]


# 5.2) Define model
model = GradientBoostingRegressor(n_estimators=300, max_depth=15, random_state=0, loss='absolute_error',
                                  learning_rate=0.04,)

# 5. Train and Test (without CV)
print("Size of the dataset:", len(features))
print("Number of features:", len(features.columns))
# Split training and testing
features_train, features_test = train_test_split(features, test_size=0.3, random_state=0, shuffle=True, stratify=targets)
targets_train, targets_test = targets[features_train.index], targets[features_test.index]



# 4) Data Augmentation in the underrepresented MMSE scores

# Histogram before
plt.hist(targets_train, bins=27, rwidth=0.8)
plt.title("Before")
plt.show()

# 4.0. Create more examples of missing targets, by interpolation of the existing ones
def interpolate_missing_mmse(F, T, missing_targets):
    print("Missing targets:", missing_targets)
    for target in missing_targets:
        # Find the closest targets
        lower_target = max([t for t in T if t < target])
        upper_target = min([t for t in T if t > target])
        # Get the features of the closest targets
        lower_features = F[T == lower_target]
        upper_features = F[T == upper_target]
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
        F = pd.concat([F, new_features])
        new_target = int((lower_target + upper_target) / 2)
        T = pd.concat([T, Series([new_target] * len(new_features), index=new_features.index)])
        print(f"Interpolated {len(new_features)} examples for target {new_target}, from targets {lower_target} and {upper_target}")

        return F, T

while True:
    min_target = targets_train.min()
    max_target = targets_train.max()
    all_targets = targets_train.unique()
    missing_targets = [i for i in range(min_target, max_target + 1) if i not in all_targets]
    if len(missing_targets) == 0:
        break
    else:
        print("New round of interpolation")
        features_train, targets_train = interpolate_missing_mmse(features_train, targets_train, missing_targets)

# Histogram after interpolation
plt.hist(targets_train, bins=27, rwidth=0.8)
plt.title("After interpolation of missing targets")
plt.show()

# 4.2. Data Augmentation method = SMOTE-C
smote = SMOTE(random_state=42, k_neighbors=3, sampling_strategy='auto')
features_train, targets_train = smote.fit_resample(features_train, targets_train)

# Histogram after
plt.hist(targets_train, bins=27, rwidth=0.8)
plt.title("After")
plt.show()

print("Features shape after DA:", features_train.shape)

# Normalise
features_train = feature_wise_normalisation(features_train, method='min-max')
features_test = feature_wise_normalisation(features_test, method='min-max')

# Train
model.fit(features_train, targets_train)

# Histogram test set
plt.hist(targets_test, bins=27, rwidth=0.8)
plt.title("Test set")
plt.show()

# Test
preditcions = model.predict(features_test)

# MAE
mae = np.mean(np.abs(targets_test - preditcions))
print("MAE:", mae)
# MSE
mse = np.mean((targets_test - preditcions) ** 2)
print("MSE:", mse)
# R2
r2 = r2_score(targets_test.to_numpy(), preditcions)
print("R2:", r2)

