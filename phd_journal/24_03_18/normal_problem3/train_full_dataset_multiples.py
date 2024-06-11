from pickle import dump

import numpy as np
from matplotlib import pyplot as plt
from pandas import Series
from sklearn.ensemble import GradientBoostingRegressor

from read import *
from read import read_all_features
from utils import feature_wise_normalisation

out_path = './scheme1'


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
FEATURES_SELECTED = ['Spectral#PeakFrequency#C3#delta', 'Spectral#Entropy#C3#theta', 'Spectral#Flatness#C3#alpha', 'Spectral#EdgeFrequency#C3#alpha', 'Spectral#PeakFrequency#C3#alpha', 'Spectral#Diff#C3#alpha', 'Spectral#Entropy#C3#beta', 'Spectral#Diff#C3#beta', 'Spectral#Entropy#C3#gamma', 'Spectral#Flatness#C3#gamma', 'Spectral#EdgeFrequency#C4#delta', 'Spectral#PeakFrequency#C4#delta', 'Spectral#Diff#C4#delta', 'Spectral#RelativePower#C4#theta', 'Spectral#Flatness#C4#theta', 'Spectral#Flatness#C4#alpha', 'Spectral#EdgeFrequency#C4#alpha', 'Spectral#PeakFrequency#C4#alpha', 'Spectral#RelativePower#C4#beta', 'Spectral#Entropy#C4#beta', 'Spectral#RelativePower#C4#gamma', 'Spectral#Flatness#C4#gamma', 'Spectral#PeakFrequency#C4#gamma', 'Spectral#Diff#C4#gamma', 'Spectral#RelativePower#Cz#delta', 'Spectral#Flatness#Cz#delta', 'Spectral#Diff#Cz#delta', 'Spectral#RelativePower#Cz#theta', 'Spectral#Flatness#Cz#theta', 'Spectral#PeakFrequency#Cz#theta', 'Spectral#Entropy#Cz#alpha', 'Spectral#EdgeFrequency#Cz#alpha', 'Spectral#RelativePower#Cz#beta', 'Spectral#EdgeFrequency#Cz#beta', 'Spectral#Flatness#Cz#gamma', 'Spectral#EdgeFrequency#Cz#gamma', 'Spectral#RelativePower#F3#delta', 'Spectral#Flatness#F3#delta', 'Spectral#RelativePower#F3#theta', 'Spectral#Entropy#F3#theta', 'Spectral#EdgeFrequency#F3#theta', 'Spectral#PeakFrequency#F3#theta', 'Spectral#Diff#F3#theta', 'Spectral#RelativePower#F3#beta', 'Spectral#EdgeFrequency#F3#beta', 'Spectral#PeakFrequency#F3#beta', 'Spectral#EdgeFrequency#F3#gamma', 'Spectral#Diff#F3#gamma', 'Spectral#RelativePower#F4#delta', 'Spectral#Diff#F4#theta', 'Spectral#PeakFrequency#F4#alpha', 'Spectral#RelativePower#F4#gamma', 'Spectral#Entropy#F4#gamma', 'Spectral#PeakFrequency#F4#gamma', 'Spectral#RelativePower#F7#delta', 'Spectral#Entropy#F7#delta', 'Spectral#Flatness#F7#delta', 'Spectral#EdgeFrequency#F7#delta', 'Spectral#Entropy#F7#theta', 'Spectral#EdgeFrequency#F7#theta', 'Spectral#Diff#F7#theta', 'Spectral#Entropy#F7#alpha', 'Spectral#Entropy#F7#beta', 'Spectral#EdgeFrequency#F7#beta', 'Spectral#PeakFrequency#F7#beta', 'Spectral#Entropy#F7#gamma', 'Spectral#EdgeFrequency#F7#gamma', 'Spectral#RelativePower#F8#delta', 'Spectral#Flatness#F8#delta', 'Spectral#EdgeFrequency#F8#delta', 'Spectral#Entropy#F8#theta', 'Spectral#EdgeFrequency#F8#theta', 'Spectral#PeakFrequency#F8#theta', 'Spectral#Diff#F8#theta', 'Spectral#RelativePower#F8#alpha', 'Spectral#Flatness#F8#alpha', 'Spectral#EdgeFrequency#F8#alpha', 'Spectral#PeakFrequency#F8#alpha', 'Spectral#RelativePower#F8#beta', 'Spectral#Entropy#F8#beta']


# 1) Get all features
features = read_all_features('KJPP', multiples=True)

# 1.1) Select features
features = features[FEATURES_SELECTED]

# drop sessions with missing values
features = features.dropna()

# remove $ from the index
features.index = features.index.str.split('$').str[0]

# 1.2) Keep only those without diagnoses
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
    if '$' in str(session):  # Multiples
        key = str(session).split('$')[0]  # remove the multiple
    else:
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

# 3) Define model
model = GradientBoostingRegressor(n_estimators=300, max_depth=15, random_state=0, loss='absolute_error',
                                  learning_rate=0.04,)

# 4) Train a model and save
train_full_dataset(model, dataset)


