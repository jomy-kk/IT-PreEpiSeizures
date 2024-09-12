from pickle import dump

import numpy as np
from matplotlib import pyplot as plt
from pandas import Series
from sklearn.ensemble import GradientBoostingRegressor

from read import *
from read import read_all_features
from utils import feature_wise_normalisation


def train_full_dataset(model, features, targets):
    print(model)
    print("Features shape:", features.shape)
    print("Targets shape:", targets.shape)

    ######################
    # SMOTE-C

    # Round targets by half-way
    targets = targets.round()
    # Discard targets less than 4
    features = features[targets >= 4]
    targets = targets[targets >= 4]

    from imblearn.over_sampling import SMOTE
    smote = SMOTE(random_state=42, k_neighbors=3, sampling_strategy='auto')

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

    # Normalise feature vectors AFTER
    features = feature_wise_normalisation(features, method='min-max')

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
    plt.xlabel('True Age (integer years)')
    plt.ylabel('Predicted Age (integer years)')
    plt.xlim(3, 19)
    plt.ylim(3, 19)
    plt.tight_layout()
    #plt.show()
    plt.savefig(join(out_path, 'train.png'))

    # 9. Serialize model
    with open(join(out_path, 'model.pkl'), 'wb') as f:
        dump(model, f)

# new: MI + RFE (932 -> 200 -> 80 features)
FEATURES_SELECTED = ['Spectral#Diff#C3#theta', 'Spectral#RelativePower#C3#gamma', 'Spectral#RelativePower#C4#delta', 'Spectral#PeakFrequency#C4#beta1', 'Spectral#Flatness#C4#gamma', 'Spectral#RelativePower#Cz#beta2', 'Spectral#RelativePower#Cz#beta3', 'Spectral#RelativePower#Cz#gamma', 'Spectral#Diff#Cz#gamma', 'Spectral#Entropy#F4#beta2', 'Spectral#Flatness#F4#beta2', 'Spectral#Entropy#F7#theta', 'Spectral#PeakFrequency#Fp2#alpha1', 'Spectral#PeakFrequency#Fp2#beta2', 'Spectral#RelativePower#Fz#delta', 'Spectral#PeakFrequency#Fz#theta', 'Spectral#PeakFrequency#Fz#gamma', 'Spectral#PeakFrequency#O1#beta3', 'Spectral#Entropy#O2#delta', 'Spectral#PeakFrequency#O2#theta', 'Spectral#PeakFrequency#P3#gamma', 'Spectral#Diff#P4#beta2', 'Spectral#EdgeFrequency#Pz#gamma', 'Spectral#EdgeFrequency#T3#delta', 'Spectral#Flatness#T5#alpha2', 'Hjorth#Activity#C3', 'Hjorth#Activity#P4', 'Hjorth#Mobility#Cz', 'PLI#Temporal(L)-Parietal(L)#alpha2', 'PLI#Temporal(L)-Occipital(L)#beta1']

# 1) Get all features
features = read_all_features('KJPP', multiples=True)
features.index = features.index.str.split('$').str[0]  # remove $ from the index

# 1.1) Select features
features = features[FEATURES_SELECTED]
features = features.dropna()

# 1.2) Keep only sessions without diagnoses
print("Number of subjects before removing outliers:", len(features))

BAD_DIAGNOSES = np.loadtxt("/Volumes/MMIS-Saraiv/Datasets/KJPP/session_ids/bad_diagnoses.txt", dtype=str)
n_before = len(features)
features = features.drop(BAD_DIAGNOSES, errors='ignore')
print("Removed Bad diagnoses:", n_before - len(features))

MAYBE_BAD_DIAGNOSES = np.loadtxt("/Volumes/MMIS-Saraiv/Datasets/KJPP/session_ids/maybe_bad_diagnoses.txt", dtype=str)
n_before = len(features)
features = features.drop(MAYBE_BAD_DIAGNOSES, errors='ignore')
print("Removed Maybe-Bad diagnoses:", n_before - len(features))

NO_REPORT = np.loadtxt("/Volumes/MMIS-Saraiv/Datasets/KJPP/session_ids/no_report.txt", dtype=str)
n_before = len(features)
features = features.drop(NO_REPORT, errors='ignore')
print("Removed No report:", n_before - len(features))
#features = features[features.index.isin(NO_REPORT)]

#NO_MEDICATION = np.loadtxt("/Volumes/MMIS-Saraiv/Datasets/KJPP/session_ids/no_medication.txt", dtype=str)
#n_before = len(features)
#features = features[features.index.isin(NO_MEDICATION)]  # keep only those with no medication
#print("Removed with medication:", n_before - len(features))

print("Number of subjects after removing outliers:", len(features))

# 2) Get targerts
targets = Series()
ages = read_ages('KJPP')
n_age_not_found = 0
for session in features.index:
    key = session
    if key in ages:
        age = ages[key]
        targets.loc[session] = age
    else:
        print(f"Session {session} not found in ages")
        n_age_not_found += 1
# Drop sessions without age
print(f"Number of sessions without age: {n_age_not_found}")
targets = targets.dropna()
features = features.loc[targets.index]

# 3) Normalise Between 0 and 1
features = feature_wise_normalisation(features, 'min-max')

# 3) Define model
model = GradientBoostingRegressor(n_estimators=300, max_depth=15, random_state=0, loss='absolute_error',
                                  learning_rate=0.04,)

# 4) Train a model and save
out_path = './scheme1'
train_full_dataset(model, features, targets)
