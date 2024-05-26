from math import floor, ceil
from math import floor, ceil
from matplotlib import pyplot as plt
from sklearn.cross_decomposition import CCA
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.model_selection import cross_val_score

from read import *


#########
# 1) Features and Targets

# 1.1) Get elders features and targets
elders_X, elders_Y = read_elders_cohorts()#select_features=SELECTED_FEATURES)
print("Elders features shape:", elders_X.shape)

# 1.2) Get children features and targets
children_X, children_Y = read_children_cohorts()#select_features=SELECTED_FEATURES)
print("Children features shape:", children_X.shape)

#########
# 2) RF: Random Forest feature importance

# 2.1) Make subset for feature selection, that is balanced in terms of age groups and associated MMSE scores.
# Select one child per each elder.
elders_X_trimmed = []
elders_Y_trimmed = []
children_X_trimmed = []
children_Y_trimmed = []
for mmse in elders_Y.unique():
    if 0 <= mmse <= 9:
        age_min, age_max = 0, 5
    elif 10 <= mmse <= 15:
        age_min, age_max = 2, 7
    elif 16 <= mmse <= 19:
        age_min, age_max = 6, 12
    elif 20 <= mmse <= 24:
        age_min, age_max = 8, 19
    elif 25 <= mmse <= 29:
        age_min, age_max = 13, 24
    elif 30 == mmse:
        age_min, age_max = 19, 24

    elders_X_sub = elders_X[elders_Y == mmse]
    n_elders = len(elders_X_sub)

    # Get the same amount of children that are in the determined age range
    children_X_sub = children_X[(children_Y >= age_min) & (children_Y <= age_max)]
    if len(children_X_sub) > n_elders:
        # sample randomly
        children_X_sub = children_X_sub.sample(n_elders)
    #children_Y_sub = children_Y[children_X_sub.index]

    if len(children_X_sub) < n_elders:
        # sample the elders
        elders_X_sub = elders_X_sub.sample(len(children_X_sub))
    elders_Y_sub = elders_Y[elders_X_sub.index]

    # let's give the children the same MMSE scores as the elders
    children_Y_sub = elders_Y[elders_X_sub.index]

    elders_X_trimmed.append(elders_X_sub)
    children_X_trimmed.append(children_X_sub)
    elders_Y_trimmed.append(elders_Y_sub)
    children_Y_trimmed.append(children_Y_sub)
# Concatenate all the data
elders_X_trimmed = pd.concat(elders_X_trimmed)
print("Trimmed Elders features shape:", elders_X_trimmed.shape)
elders_Y_trimmed = pd.concat(elders_Y_trimmed)
children_X_trimmed = pd.concat(children_X_trimmed)
print("Trimmed Children features shape:", children_X_trimmed.shape)
children_Y_trimmed = pd.concat(children_Y_trimmed)

# Make one single dataset
X_fs = pd.concat([elders_X_trimmed, children_X_trimmed])
Y_fs = pd.concat([elders_Y_trimmed, children_Y_trimmed])

# 2.2) Fit an RF model
model_fs = RandomForestRegressor(n_estimators=200, max_depth=10, random_state=0)
model_fs.fit(X_fs, Y_fs)

# 2.3) Get feature importances
feature_importances = model_fs.feature_importances_
# Get 80 most important features
n_features = 20
feature_indices = np.argsort(feature_importances)[::-1][:n_features]
selected_features = X_fs.columns[feature_indices]
print("Selected features:", selected_features)
# Plot feature importances
plt.figure()
plt.barh(range(n_features), feature_importances[feature_indices], align='center')
plt.yticks(range(n_features), selected_features)
plt.xlabel('Feature importance')
plt.ylabel('Feature')
plt.show()

#########

# 3) Select features
elders_X = elders_X[selected_features]
elders_X_trimmed = elders_X_trimmed[selected_features]
print("Elders features shape:", elders_X.shape)
children_X = children_X[selected_features]
children_X_trimmed = children_X_trimmed[selected_features]
print("Children features shape:", children_X.shape)

# 3.1) Fit CCA
cca = CCA(n_components=20)
cca.fit(elders_X_trimmed, children_X_trimmed)
print("CCA fitted")

# 3.2) Transform the whole datasets
transformed_elders_X, transformed_children_X = cca.transform(elders_X, children_X)
print("Transformed Elders features shape:", transformed_elders_X.shape)
print("Transformed Children features shape:", transformed_children_X.shape)


#########

# 4.1) Model
model = GradientBoostingRegressor(n_estimators=200, max_depth=10, random_state=0, loss='absolute_error',
                                  learning_rate=0.04,)

# 4.2) Train
model.fit(elders_X, elders_Y)
print("Gradient Boosting Regressor trained")

# 4.3) Validate
def validate(model, features, targets):
    # 4.2) Estimates on children
    predictions = model.predict(features)

    def is_good_developmental_age_estimate(age: float, mmse: int, margin:float=0) -> bool:
        """
        Checks if the MMSE estimate is within the acceptable range for the given age.
        A margin can be added to the acceptable range.
        """
        assert 0 <= mmse <= 30, "MMSE must be between 0 and 30"
        assert 0 <= age, "Developmental age estimate must be positive"

        if age < 1.25:
            return 0 - margin <= mmse <= age / 2 + margin
        elif age < 2:
            return floor((4 * age / 15) - (1 / 3)) - margin <= mmse <= ceil(age / 2) + margin
        elif age < 5:
            return (4 * age / 15) - (1 / 3) - margin <= mmse <= 2 * age + 5 + margin
        elif age < 7:
            return 2 * age - 6 - margin <= mmse <= (4 * age / 3) + (25 / 3) + margin
        elif age < 8:
            return (4 * age / 5) + (47 / 5) - margin <= mmse <= (4 * age / 3) + (25 / 3) + margin
        elif age < 12:
            return (4 * age / 5) + (47 / 5) - margin <= mmse <= (4 * age / 5) + (68 / 5) + margin
        elif age < 13:
            return (4 * age / 7) + (92 / 7) - margin <= mmse <= (4 * age / 5) + (68 / 5) + margin
        elif age < 19:
            return (4 * age / 7) + (92 / 7) - margin <= mmse <= 30 + margin
        elif age >= 25:
            return mmse >= 29 - margin

    # 6) Plot
    accurate = []
    inaccurate = []
    for prediction, age in zip(predictions, targets):
        if is_good_developmental_age_estimate(age, prediction, margin=1.5):
            accurate.append((age, prediction))
        else:
            inaccurate.append((age, prediction))

    accurate_x, accurate_y = zip(*accurate)
    inaccurate_x, inaccurate_y = zip(*inaccurate)

    # 9. Plot predictions vs targets
    plt.figure()
    plt.ylabel('Estimate')
    plt.xlabel('Ground Truth')
    plt.xlim(0, 30)
    #plt.xlim(0, 22)
    plt.grid(linestyle='--', alpha=0.4)
    plt.scatter(accurate_x, accurate_y, color="#34AC8B", marker='.', alpha=0.5)
    plt.scatter(inaccurate_x, inaccurate_y, color='red', marker='.', alpha=0.5)
    # remove box around plot
    plt.box(False)
    plt.show()

    # 10. Metrics

    # Percentage right
    percentage_right = len(accurate) / (len(accurate) + len(inaccurate))
    print("Correct Bin Assignment:", percentage_right)

    # R2 Score
    from sklearn.metrics import r2_score
    # Normalize between 0 and 1
    targets_norm = (targets - targets.min()) / (targets.max() - targets.min())
    predictions_norm = (predictions - predictions.min()) / (predictions.max() - predictions.min())
    r2 = r2_score(targets_norm, predictions_norm)
    print("R2 Score:", r2)

    # pearson rank correlation
    from scipy.stats import pearsonr
    pearson, pvalue = pearsonr(targets, predictions)
    print("Pearson rank correlation:", pearson, f"(p={pvalue})")

    # Spearman rank correlation
    from scipy.stats import spearmanr
    spearman, pvalue = spearmanr(targets, predictions, alternative='greater')
    print("Spearman rank correlation:", spearman, f"(p={pvalue})")

    # Kendal rank correlation
    from scipy.stats import kendalltau
    kendall, pvalue = kendalltau(targets, predictions, alternative='greater')
    print("Kendall rank correlation:", kendall, f"(p={pvalue})")

    # Somers' D
    from scipy.stats import somersd
    res = somersd(targets, predictions)
    correlation, pvalue, table = res.statistic, res.pvalue, res.table
    print("Somers' D:", correlation, f"(p={pvalue})")

    # Confusion Matrix
    # We'll have 4 classes
    # here are the boundaries
    age_classes = ((0, 5), (5, 8), (8, 13), (13, 25))
    mmse_classes = ((0, 9), (9, 15), (15, 24), (24, 30))

    # assign predictions to classes
    mmse_classes_assigned = []
    for prediction in predictions:
        for i, (lower, upper) in enumerate(mmse_classes):
            if lower <= float(prediction) <= upper:
                mmse_classes_assigned.append(i)
                break
    # assign targets to classes
    age_classes_assigned = []
    for age in targets:
        for i, (lower, upper) in enumerate(age_classes):
            if lower <= age <= upper:
                age_classes_assigned.append(i)
                break


validate(model, children_X, children_Y)
