import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from pandas import Series
from sklearn.cross_decomposition import CCA
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.feature_selection import RFE, VarianceThreshold, SelectKBest, SelectPercentile, r_regression, f_regression, \
    mutual_info_regression, RFECV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split, cross_val_score

from read import *
from read import read_all_features
from utils import feature_wise_normalisation


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


#########
# 1) Features and Targets

# 1.1) Get elders features and targets
elders_X, elders_Y = read_elders_cohorts(miltiadous_multiples=False)
print("Elders features shape:", elders_X.shape)

# 1.2) Get children features and targets
children_X, children_Y = read_children_cohorts()
print("Children features shape:", children_X.shape)

#########
# 2) CCA: Canonical Correlation Analysis

# 2.1) Sort elders_X by MMSE (elders_Y)
elders_Y = elders_Y.sort_values()
elders_X = elders_X.loc[elders_Y.index]

# 2.2) Sort children_X by MMSE (children_Y)
children_Y = children_Y.sort_values()
children_X = children_X.loc[children_Y.index]

# 2.3) Fit CCA
cca = CCA(n_components=10)
cca.fit(elders_X, children_X)

# 2.4) Transform features
transformed_elders_X, transformed_children_X = cca.transform(elders_X, children_X)

print("Relevant features", transformed_elders_X.columns)

exit(0)






# 8. Get metrics for train set
model.fit(transformed_features, targets)
predictions = model.predict(transformed_features)
print('---------------------------------')
print('Train set metrics:')
mse = mean_squared_error(targets, predictions)
print(f'MSE: {mse}')
mae = mean_absolute_error(targets, predictions)
print(f'MAE: {mae}')
r2 = r2_score(targets, predictions)
print(f'R2: {r2}')
# 8.1. Plot regression between ground truth and predictions with seaborn and draw regression curve
plt.figure(figsize=((3.5,3.2)))
sns.regplot(x=targets, y=predictions, scatter_kws={'alpha': 0.4}, color="#34AC8B")
plt.xlabel('True MMSE (units)')
plt.ylabel('Predicted MMSE (units)')
plt.xlim(0, 30)
plt.ylim(0, 30)
plt.tight_layout()
plt.show()

#########
# VALIDATION

# 9. Update dataset_val
dataset_val = [([y for i, y in enumerate(x[0]) if i in indices], x[1]) for x in dataset_val]
objects = np.array([x[0] for x in dataset_val])
targets = np.array([x[1] for x in dataset_val])

# 10. Make predictions
print("Size of validation dataset:", len(dataset_val))
print("Number of features:", len(dataset_val[0][0]))
predictions = model.predict(transformed_features)

# 11. Get metrics for validation set
print('---------------------------------')
print('Validation set metrics:')
mse = mean_squared_error(targets, predictions)
print(f'MSE: {mse}')
mae = mean_absolute_error(targets, predictions)
print(f'MAE: {mae}')
r2 = r2_score(targets, predictions)
print(f'R2: {r2}')
# 8.1. Plot regression between ground truth and predictions with seaborn and draw regression curve
plt.figure(figsize=((3.5,3.2)))
sns.regplot(x=targets, y=predictions, scatter_kws={'alpha': 0.4}, color="#34AC8B")
plt.xlabel('True MMSE (units)')
plt.ylabel('Predicted MMSE (units)')
plt.xlim(0, 30)
plt.ylim(0, 30)
plt.tight_layout()
plt.show()


