import numpy as np
from matplotlib import pyplot as plt
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split, cross_val_score, KFold

from ..read import *
from ..read import read_all_features
from ..utils import feature_wise_normalisation


def feature_importances(model):
    importances = model.feature_importances_
    importances = pd.Series(importances, index=feature_names)
    importances = importances.nlargest(20) # Get max 20 features
    fig, ax = plt.subplots()
    importances.plot.bar(ax=ax)
    ax.set_title("Feature importances using MDI")
    ax.set_ylabel("Mean decrease in impurity")
    fig.tight_layout()
    plt.show()


def train_test_cv(model, cv, objects, targets):
    scores = cross_val_score(model, objects, targets,
                             cv=cv, scoring='r2', #'neg_mean_absolute_error',
                             verbose=2, n_jobs=-1)
    print("Cross-Validation mean score:", scores.mean())
    print("Cross-Validation std score:", scores.std())
    print("Cross-Validation max score:", scores.max())
    print("Cross-Validation min score:", scores.min())


def train_test(model, train_size, random_state):
    print(model)

    # 4) Split subjects into train and test (using sklearn)
    train_dataset, test_dataset = train_test_split(dataset, train_size=train_size, shuffle=True, random_state=random_state)

    # Separate features and targets
    train_features = np.array([x[0] for x in train_dataset])
    train_targets = np.array([x[1] for x in train_dataset])
    print("Train features shape:", train_features.shape)
    print("Train targets shape:", train_targets.shape)
    test_features = np.array([x[0] for x in test_dataset])
    test_targets = np.array([x[1] for x in test_dataset])
    print("Test features shape:", test_features.shape)
    print("Test targets shape:", test_targets.shape)

    # 5. Train the model only with the selected features
    model.fit(train_features, train_targets)

    # 6. Test the model
    predictions = model.predict(test_features)
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    # 6.1) Mean Squared Error (MSE)
    mse = mean_squared_error(test_targets, predictions)
    # 6.2) Mean Absolute Error (MAE)
    mae = mean_absolute_error(test_targets, predictions)
    # 6.3) R2 Score
    r2 = r2_score(test_targets, predictions)

    # 7. Print results
    print(f'MSE: {mse}')
    print(f'MAE: {mae}')
    print(f'R2: {r2}')
    print('---------------------------------')

    # 8. Plot regression between ground truth and predictions with seaborn and draw regression curve
    import seaborn as sns
    plt.figure(figsize=((3.5,3.2)))
    sns.regplot(x=test_targets, y=predictions, scatter_kws={'alpha': 0.4})
    #plt.title(str(model))
    plt.xlabel('True Age (years)')
    plt.ylabel('Predicted Age (years)')
    plt.xlim(4, 18)
    plt.ylim(4, 18)
    plt.tight_layout()
    plt.show()


    """
    # 8. Plot regression with one feature
    X, y = test_features[:, 60], predictions  # just first feature
    lw = 2
    plt.scatter(X, test_targets, color='darkorange', label='Target')
    plt.scatter(X, y, color='navy', lw=lw, label='Predicted')
    plt.xlabel('Hjorth Mobility F4')
    plt.ylabel('Preditcion / Target')
    plt.title(str(model))
    plt.legend()
    plt.show()
    """


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
OUTLIERS = [  8,  40,  59, 212, 229, 247, 264, 294, 309, 356, 388, 391, 429, 437, 448, 460, 465, 512, 609, 653, 687, 688, 771, 808, 831, 872, 919]
features = features.drop(features.index[OUTLIERS])
print("Number of subjects after removing outliers:", len(features))

# Normalise feature vectors
features = feature_wise_normalisation(features, method='min-max')
features = features.dropna(axis=1)

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

# 3.1) Define CV scheme
#cv = KFold(n_splits=5, shuffle=True, random_state=0)
#cv = ShuffleSplit(n_splits=10, test_size=0.1, random_state=0)
cv = KFold(10, shuffle=True)  # leave 10% out, non-overlapping test sets

# 3.2) Define model
#model = RandomForestRegressor(n_estimators=200, criterion='absolute_error', max_depth=10, random_state=0)
#model = Lasso(alpha=0.001)
model = GradientBoostingRegressor(n_estimators=200, max_depth=10, random_state=0, loss='absolute_error',
                                  learning_rate=0.04,)
#SVR(kernel='rbf', C=3, epsilon=0.1, gamma='scale')
#ElasticNet(alpha=0.1, l1_ratio=0.7)
#RandomForestRegressor(n_estimators=100, max_depth=20, random_state=0)
#BayesianRidge()
#Lasso(alpha=0.1)
#Ridge(alpha=0.1)
"""
model = MLPRegressor(hidden_layer_sizes=(32,32,32, 32, 64, 128), activation='relu',
                    alpha=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, momentum=0.9,
                    solver='adam',  batch_size='auto', validation_fraction=0.1,
                    learning_rate='adaptive', learning_rate_init=0.001, power_t=0.5, shuffle=True,
                    max_iter=500, tol=0.0001, early_stopping=False, n_iter_no_change=10, max_fun=15000,
                    verbose=True)
"""

# 5. Cross-validation results
print("Size of the dataset:", len(dataset))
print("Number of features:", len(dataset[0][0]))
objects = np.array([x[0] for x in dataset])
targets = np.array([x[1] for x in dataset])
train_test_cv(model, cv, objects, targets)

# 6. Train and Test
train_test(model, train_size=0.9, random_state=1)



