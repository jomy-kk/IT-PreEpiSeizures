from pickle import dump

import numpy as np
from matplotlib import pyplot as plt
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split, cross_val_score, KFold

from read import *
from read import read_all_features
from utils import feature_wise_normalisation


def train_full_dataset(model, dataset, random_state):
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
    plt.figure(figsize=((3.5,3.2)))
    sns.regplot(x=targets, y=predictions, scatter_kws={'alpha': 0.4})
    #plt.title(str(model))
    plt.xlabel('True Age (years)')
    plt.ylabel('Predicted Age (years)')
    plt.xlim(2, 20)
    plt.ylim(2, 20)
    plt.tight_layout()
    plt.show()

    # 9. Serialize model
    with open('model.pkl', 'wb') as f:
        dump(model, f)


# 1) Get all features
features = read_all_features('KJPP')

# Drop subject_sessions with nans
features = features.dropna()

# 1.1.) Select features
# FIXME
FEATURES_SELECTED = ['Spectral#Diff#C3#theta', 'Spectral#RelativePower#C3#gamma', 'Spectral#RelativePower#C4#delta', 'Spectral#PeakFrequency#C4#beta1', 'Spectral#Flatness#C4#gamma', 'Spectral#RelativePower#Cz#beta2', 'Spectral#RelativePower#Cz#beta3', 'Spectral#RelativePower#Cz#gamma', 'Spectral#Diff#Cz#gamma', 'Spectral#Entropy#F4#beta2', 'Spectral#Flatness#F4#beta2', 'Spectral#Entropy#F7#theta', 'Spectral#PeakFrequency#Fp2#alpha1', 'Spectral#PeakFrequency#Fp2#beta2', 'Spectral#RelativePower#Fz#delta', 'Spectral#PeakFrequency#Fz#theta', 'Spectral#PeakFrequency#Fz#gamma', 'Spectral#PeakFrequency#O1#beta3', 'Spectral#Entropy#O2#delta', 'Spectral#PeakFrequency#O2#theta', 'Spectral#PeakFrequency#P3#gamma', 'Spectral#Diff#P4#beta2', 'Spectral#EdgeFrequency#Pz#gamma', 'Spectral#EdgeFrequency#T3#delta', 'Spectral#Flatness#T5#alpha2', 'Hjorth#Activity#C3', 'Hjorth#Activity#P4', 'Hjorth#Mobility#Cz', 'PLI#Temporal(L)-Parietal(L)#alpha2', 'PLI#Temporal(L)-Occipital(L)#beta1']
features = features[FEATURES_SELECTED]
print("Number of features selected:", len(features.columns))

# 1.2.) Remove outliers
# FIXME
print("Number of subjects before removing outliers:", len(features))
OUTLIERS = [8,  40,  59, 212, 229, 247, 264, 294, 309, 356, 388, 391, 429, 437, 448, 460, 465, 512, 609, 653, 687, 688, 771, 808, 831, 872, 919]
features = features.drop(features.index[OUTLIERS])
print("Number of subjects after removing outliers:", len(features))

# Normalise feature vectors
features = feature_wise_normalisation(features, method='mean-std')
features = features.dropna(axis=1)

# Save stochastic pattern
stochastics = DataFrame([features.mean(), features.std(), features.min(), features.max()], index=['mean', 'std', 'min', 'max'])
stochastics.to_csv('kjpp_stochastic_pattern.csv')

# 2.1) Convert features to an appropriate format
feature_names = features.columns.to_numpy()
sessions = features.index.to_numpy()
features = [features.loc[code].to_numpy() for code in sessions]

# 2.2) Associate targets to features
dataset = []
ages = read_ages('KJPP')
for session, session_features in zip(sessions, features):
    age = ages[session]
    dataset.append((session_features, age))

# Save adult stochastic pattern (age>17.5)
adult_features = [x[0] for x in dataset if x[1] > 17.5]
adult_features = np.array(adult_features)
adult_stochastics = DataFrame([adult_features.mean(axis=0),
                               adult_features.std(axis=0),
                               adult_features.min(axis=0),
                               adult_features.max(axis=0)],
                              index=['mean', 'std', 'min', 'max'],
                              columns=feature_names)
adult_stochastics.to_csv('kjpp_adult_stochastic_pattern.csv')

exit(0)

# 3) Define model
model = GradientBoostingRegressor(n_estimators=200, max_depth=10, random_state=0, loss='absolute_error',
                                  learning_rate=0.04,)

# 4) Train a model and save
train_full_dataset(model, dataset, random_state=0)


