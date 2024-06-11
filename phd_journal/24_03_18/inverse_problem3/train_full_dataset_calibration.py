from pickle import dump

import numpy as np
from matplotlib import pyplot as plt
from pandas import Series
from sklearn.ensemble import GradientBoostingRegressor

from read import *
from read import read_all_features
from utils import feature_wise_normalisation

out_path = './scheme27'

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
    plt.figure()
    plt.ylabel('MMSE Estimate', fontsize=12)
    plt.xlabel('True MMSE (units)', fontsize=12)
    plt.xlim(0, 30)
    plt.ylim(0, 30)
    plt.grid(linestyle='--', alpha=0.4)
    sns.regplot(x=targets, y=predictions, scatter_kws={'alpha': 0.3, 'color': '#C60E4F'}, line_kws={'color': '#C60E4F'})
    plt.xticks([4, 6, 9, 12, 15, 20, 25, 30], fontsize=12)
    plt.yticks([4, 6, 9, 12, 15, 20, 25, 30], fontsize=12)
    plt.tight_layout()
    plt.box(False)
    #plt.show()
    plt.savefig(join(out_path, 'train.pdf'))

    # 9. Serialize model
    with open(join(out_path, 'model.pkl'), 'wb') as f:
        dump(model, f)


# 1) Get all features
insight = read_all_features('INSIGHT')
brainlat = read_all_features('BrainLat')
miltiadous = read_all_features('Miltiadous Dataset')
sapienza = read_all_features('Sapienza')
features = pd.concat([insight, brainlat, miltiadous, sapienza], axis=0)
print("Read all features. Shape:", features.shape)

# 1.1.) Select features
# 80 features from elders RFE
FEATURES_SELECTED = ['Spectral#RelativePower#C3#beta1', 'Spectral#EdgeFrequency#C3#beta3', 'Spectral#RelativePower#C3#gamma', 'Spectral#EdgeFrequency#C4#alpha1', 'Spectral#RelativePower#C4#beta3', 'Spectral#EdgeFrequency#C4#beta3', 'Spectral#EdgeFrequency#C4#gamma', 'Spectral#Flatness#Cz#theta', 'Spectral#PeakFrequency#Cz#theta', 'Spectral#EdgeFrequency#Cz#beta3', 'Spectral#EdgeFrequency#Cz#gamma', 'Spectral#PeakFrequency#Cz#gamma', 'Spectral#RelativePower#F3#beta1', 'Spectral#Diff#F4#delta', 'Spectral#RelativePower#F7#beta3', 'Spectral#EdgeFrequency#F7#beta3', 'Spectral#RelativePower#F7#gamma', 'Spectral#RelativePower#F8#beta1', 'Spectral#EdgeFrequency#F8#beta3', 'Spectral#RelativePower#Fp1#beta1', 'Spectral#EdgeFrequency#Fp1#beta3', 'Spectral#Diff#Fp2#delta', 'Spectral#RelativePower#Fp2#beta1', 'Spectral#RelativePower#Fp2#beta3', 'Spectral#Diff#Fpz#beta2', 'Spectral#Entropy#O1#delta', 'Spectral#RelativePower#O1#beta2', 'Spectral#EdgeFrequency#O1#beta2', 'Spectral#EdgeFrequency#O1#beta3', 'Spectral#RelativePower#O2#delta', 'Spectral#PeakFrequency#O2#alpha1', 'Spectral#RelativePower#O2#beta1', 'Spectral#RelativePower#O2#beta3', 'Spectral#Diff#P3#beta1', 'Spectral#RelativePower#P3#beta3', 'Spectral#RelativePower#Pz#alpha1', 'Spectral#EdgeFrequency#Pz#beta3', 'Spectral#RelativePower#T4#alpha1', 'Spectral#RelativePower#T4#beta3', 'Spectral#RelativePower#T4#gamma', 'Spectral#EdgeFrequency#T5#beta2', 'Hjorth#Complexity#T5', 'Hjorth#Complexity#P4', 'Hjorth#Complexity#F7', 'Hjorth#Complexity#T4', 'Hjorth#Complexity#F8', 'Hjorth#Complexity#T3', 'Hjorth#Mobility#P3', 'PLI#Frontal(L)-Temporal(R)#alpha1', 'PLI#Frontal(L)-Occipital(L)#alpha1', 'PLI#Frontal(R)-Temporal(R)#alpha1', 'PLI#Temporal(R)-Parietal(R)#alpha1', 'PLI#Temporal(R)-Occipital(L)#alpha1', 'PLI#Parietal(R)-Occipital(L)#alpha1', 'PLI#Occipital(L)-Occipital(R)#alpha1', 'PLI#Temporal(R)-Occipital(R)#alpha2', 'PLI#Parietal(R)-Occipital(L)#alpha2', 'COH#Frontal(L)-Frontal(R)#theta', 'COH#Frontal(L)-Occipital(L)#theta', 'COH#Frontal(L)-Occipital(R)#alpha1', 'COH#Frontal(R)-Occipital(L)#alpha1', 'COH#Parietal(R)-Occipital(L)#alpha1', 'COH#Frontal(L)-Frontal(R)#alpha2', 'COH#Frontal(L)-Occipital(R)#alpha2', 'COH#Parietal(R)-Occipital(L)#alpha2', 'COH#Parietal(R)-Occipital(R)#alpha2', 'COH#Occipital(L)-Occipital(R)#alpha2', 'COH#Frontal(L)-Occipital(L)#beta1', 'COH#Temporal(R)-Parietal(R)#beta1', 'COH#Parietal(R)-Occipital(R)#beta1', 'COH#Frontal(L)-Parietal(L)#beta2', 'COH#Frontal(R)-Occipital(L)#beta2', 'COH#Frontal(L)-Temporal(R)#beta3', 'COH#Frontal(L)-Parietal(L)#beta3', 'COH#Frontal(L)-Occipital(L)#beta3', 'COH#Frontal(L)-Occipital(R)#beta3', 'COH#Frontal(R)-Occipital(L)#beta3', 'COH#Temporal(L)-Occipital(R)#beta3', 'COH#Frontal(L)-Occipital(R)#gamma', 'COH#Frontal(R)-Occipital(R)#gamma']
features = features[FEATURES_SELECTED]
print("Number of features selected:", len(features.columns))
features = features.dropna(axis=0)

# 2) Get targets
insight_targets = read_mmse('INSIGHT')
brainlat_targets = read_mmse('BrainLat')
miltiadous_targets = read_mmse('Miltiadous Dataset')
sapienza_targets = read_mmse('Sapienza')
targets = Series()
batch = []
for index in features.index:
    if '_' in str(index):  # insight
        key = int(index.split('_')[0])
        if key in insight_targets:
            targets.loc[index] = insight_targets[key]
            batch.append(1)
    elif '-' in str(index):  # brainlat
        if index in brainlat_targets:
            targets.loc[index] = brainlat_targets[index]
            batch.append(2)
    elif 'PARTICIPANT' in str(index):  # sapienza
        if index in sapienza_targets:
            targets.loc[index] = sapienza_targets[index]
            batch.append(3)
    else:  # miltiadous
        # parse e.g. 24 -> 'sub-024'; 1 -> 'sub-001'
        if '$' in str(index):  # EXTRA: multiple examples, remove the $ and the number after it; the target is the same
            key = 'sub-' + str(str(index).split('$')[0]).zfill(3)
        else:
            key = 'sub-' + str(index).zfill(3)
        if key:
            targets.loc[index] = miltiadous_targets[key]
            batch.append(4)

# Drop subject_sessions with nans targets
targets = targets.dropna()
features = features.loc[targets.index]

# Normalise feature vectors BEFORE
features = feature_wise_normalisation(features, method='min-max')
features = features.dropna(axis=1)


mmse_scores = sorted(list(set(targets)))
mmse_distribution = [len(targets[targets == mmse]) for mmse in mmse_scores]
print("MMSE distribution before augmentation:")
for i, mmse in enumerate(mmse_scores):
    print(f"MMSE {mmse}: {mmse_distribution[i]} examples")

"""
######################
# SMOTE DATA AUGMENTATION

from imblearn.over_sampling import SMOTE

# let's make target 6 -> 4, and 12 -> 9, and 15->16
targets = targets.replace(6, 4)
targets = targets.replace(12, 9)
targets = targets.replace(15, 16)
smote = SMOTE(random_state=42, k_neighbors=1, sampling_strategy='auto')
features, targets = smote.fit_resample(features, targets)
######################
"""

# Normalise feature vectors AFTER
features = feature_wise_normalisation(features, method='min-max')
features = features.dropna(axis=1)

# Save normalised features and targets
features.to_csv(join(out_path, 'elders_features.csv'))
targets.to_csv(join(out_path, 'elders_targets.csv'))


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


