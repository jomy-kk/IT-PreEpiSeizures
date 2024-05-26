from math import floor, ceil
from math import floor, ceil
from matplotlib import pyplot as plt
from sklearn.cross_decomposition import CCA
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import cross_val_score

from read import *


SELECTED_FEATURES_ELDERS = {'Spectral#RelativePower#C3#beta1', 'Spectral#EdgeFrequency#C3#beta3',
                            'Spectral#RelativePower#C3#gamma', 'Spectral#EdgeFrequency#C4#alpha1',
                            'Spectral#RelativePower#C4#beta3', 'Spectral#EdgeFrequency#C4#beta3',
                            'Spectral#EdgeFrequency#C4#gamma', 'Spectral#Flatness#Cz#theta',
                            'Spectral#PeakFrequency#Cz#theta', 'Spectral#EdgeFrequency#Cz#beta3',
                            'Spectral#EdgeFrequency#Cz#gamma', 'Spectral#PeakFrequency#Cz#gamma',
                            'Spectral#RelativePower#F3#beta1', 'Spectral#Diff#F4#delta',
                            'Spectral#RelativePower#F7#beta3', 'Spectral#EdgeFrequency#F7#beta3',
                            'Spectral#RelativePower#F7#gamma', 'Spectral#RelativePower#F8#beta1',}
"""
                            'Spectral#EdgeFrequency#F8#beta3', 'Spectral#RelativePower#Fp1#beta1',
                            'Spectral#EdgeFrequency#Fp1#beta3', 'Spectral#Diff#Fp2#delta',
                            'Spectral#RelativePower#Fp2#beta1', 'Spectral#RelativePower#Fp2#beta3',
                            'Spectral#Diff#Fpz#beta2', 'Spectral#Entropy#O1#delta', 'Spectral#RelativePower#O1#beta2',
                            'Spectral#EdgeFrequency#O1#beta2', 'Spectral#EdgeFrequency#O1#beta3',
                            'Spectral#RelativePower#O2#delta', 'Spectral#PeakFrequency#O2#alpha1',
                            'Spectral#RelativePower#O2#beta1', 'Spectral#RelativePower#O2#beta3',
                            'Spectral#Diff#P3#beta1', 'Spectral#RelativePower#P3#beta3',
                            'Spectral#RelativePower#Pz#alpha1', 'Spectral#EdgeFrequency#Pz#beta3',
                            'Spectral#RelativePower#T4#alpha1', 'Spectral#RelativePower#T4#beta3',
                            'Spectral#RelativePower#T4#gamma', 'Spectral#EdgeFrequency#T5#beta2',
                            'Hjorth#Complexity#T5', 'Hjorth#Complexity#P4', 'Hjorth#Complexity#F7',
                            'Hjorth#Complexity#T4', 'Hjorth#Complexity#F8', 'Hjorth#Complexity#T3',
                            'Hjorth#Mobility#P3', 'PLI#Frontal(L)-Temporal(R)#alpha1',
                            'PLI#Frontal(L)-Occipital(L)#alpha1', 'PLI#Frontal(R)-Temporal(R)#alpha1',
                            'PLI#Temporal(R)-Parietal(R)#alpha1', 'PLI#Temporal(R)-Occipital(L)#alpha1',
                            'PLI#Parietal(R)-Occipital(L)#alpha1', 'PLI#Occipital(L)-Occipital(R)#alpha1',
                            'PLI#Temporal(R)-Occipital(R)#alpha2', 'PLI#Parietal(R)-Occipital(L)#alpha2',
                            'COH#Frontal(L)-Frontal(R)#theta', 'COH#Frontal(L)-Occipital(L)#theta',
                            'COH#Frontal(L)-Occipital(R)#alpha1', 'COH#Frontal(R)-Occipital(L)#alpha1',
                            'COH#Parietal(R)-Occipital(L)#alpha1', 'COH#Frontal(L)-Frontal(R)#alpha2',
                            'COH#Frontal(L)-Occipital(R)#alpha2', 'COH#Parietal(R)-Occipital(L)#alpha2',
                            'COH#Parietal(R)-Occipital(R)#alpha2', 'COH#Occipital(L)-Occipital(R)#alpha2',
                            'COH#Frontal(L)-Occipital(L)#beta1', 'COH#Temporal(R)-Parietal(R)#beta1',
                            'COH#Parietal(R)-Occipital(R)#beta1', 'COH#Frontal(L)-Parietal(L)#beta2',
                            'COH#Frontal(R)-Occipital(L)#beta2', 'COH#Frontal(L)-Temporal(R)#beta3',
                            'COH#Frontal(L)-Parietal(L)#beta3', 'COH#Frontal(L)-Occipital(L)#beta3',
                            'COH#Frontal(L)-Occipital(R)#beta3', 'COH#Frontal(R)-Occipital(L)#beta3',
                            'COH#Temporal(L)-Occipital(R)#beta3', 'COH#Frontal(L)-Occipital(R)#gamma',
                            'COH#Frontal(R)-Occipital(R)#gamma'}
"""
SELECTED_FEATURES_CHILDREN = {'Spectral#EdgeFrequency#C4#theta', 'Spectral#PeakFrequency#C4#gamma',
                              'Spectral#RelativePower#Cz#gamma', 'Spectral#RelativePower#F3#delta',
                              'Spectral#RelativePower#F3#alpha1', 'Spectral#PeakFrequency#F7#beta3',
                              'Spectral#EdgeFrequency#F8#gamma', 'Spectral#RelativePower#Fp1#delta',
                              'Spectral#RelativePower#Fp1#alpha1', 'Spectral#Entropy#Fp2#beta3',
                              'Spectral#RelativePower#Fz#alpha1', 'Spectral#PeakFrequency#O1#alpha1',
                              'Spectral#EdgeFrequency#O1#beta3', 'Spectral#RelativePower#O2#theta',
                              'Spectral#Entropy#O2#beta2', 'Spectral#Flatness#O2#beta2',
                              'Spectral#EdgeFrequency#O2#beta3', 'Spectral#PeakFrequency#O2#gamma',}
"""
                              'Spectral#EdgeFrequency#P3#gamma', 'Spectral#PeakFrequency#P4#alpha1',
                              'Spectral#EdgeFrequency#P4#gamma', 'Spectral#PeakFrequency#Pz#alpha1',
                              'Spectral#EdgeFrequency#T4#beta2', 'Spectral#RelativePower#T5#beta2',
                              'Spectral#Flatness#T5#beta3', 'Hjorth#Activity#P4', 'Hjorth#Activity#F4',
                              'Hjorth#Activity#C4', 'Hjorth#Activity#F8', 'Hjorth#Activity#C3', 'Hjorth#Mobility#O2',
                              'Hjorth#Mobility#T3', 'Hjorth#Mobility#Fz', 'Hjorth#Mobility#Cz', 'Hjorth#Mobility#T6',
                              'Hjorth#Mobility#Fp1', 'Hjorth#Mobility#C4', 'Hjorth#Mobility#P3', 'Hjorth#Mobility#F8',
                              'Hjorth#Mobility#O1', 'Hjorth#Mobility#T4', 'Hjorth#Complexity#O2',
                              'Hjorth#Complexity#F3', 'Hjorth#Complexity#Fz', 'PLI#Temporal(L)-Occipital(R)#alpha1',
                              'PLI#Parietal(L)-Occipital(L)#alpha1', 'COH#Frontal(L)-Frontal(R)#delta',
                              'COH#Frontal(L)-Parietal(R)#delta', 'COH#Temporal(L)-Parietal(R)#delta',
                              'COH#Temporal(R)-Parietal(L)#delta', 'COH#Temporal(R)-Parietal(R)#delta',
                              'COH#Temporal(L)-Occipital(L)#theta', 'COH#Temporal(R)-Parietal(R)#theta',
                              'COH#Temporal(R)-Occipital(L)#theta', 'COH#Temporal(R)-Occipital(R)#theta',
                              'COH#Frontal(L)-Temporal(R)#alpha1', 'COH#Frontal(L)-Parietal(R)#alpha1',
                              'COH#Frontal(L)-Occipital(L)#alpha1', 'COH#Frontal(R)-Temporal(L)#alpha1',
                              'COH#Frontal(R)-Parietal(L)#alpha1', 'COH#Temporal(L)-Parietal(R)#alpha1',
                              'COH#Frontal(L)-Parietal(R)#alpha2', 'COH#Frontal(L)-Occipital(L)#alpha2',
                              'COH#Frontal(R)-Parietal(L)#alpha2', 'COH#Frontal(R)-Occipital(R)#alpha2',
                              'COH#Frontal(L)-Occipital(L)#beta1', 'COH#Frontal(R)-Parietal(L)#beta1',
                              'COH#Frontal(R)-Occipital(R)#beta1', 'COH#Temporal(L)-Parietal(L)#beta1',
                              'COH#Temporal(R)-Parietal(L)#beta1', 'COH#Temporal(R)-Parietal(R)#beta1',
                              'COH#Frontal(L)-Frontal(R)#beta2', 'COH#Frontal(L)-Temporal(L)#beta2',
                              'COH#Frontal(L)-Parietal(R)#beta2', 'COH#Frontal(L)-Occipital(L)#beta2',
                              'COH#Frontal(R)-Occipital(R)#beta2', 'COH#Frontal(L)-Parietal(R)#beta3',
                              'COH#Frontal(L)-Occipital(L)#beta3', 'COH#Frontal(R)-Parietal(L)#beta3',
                              'COH#Frontal(R)-Occipital(L)#gamma'}
"""
SELECTED_FEATURES = list(SELECTED_FEATURES_CHILDREN.union(SELECTED_FEATURES_ELDERS))
print("Union of Selected features:", len(SELECTED_FEATURES))

#########
# 1) Features and Targets

# 1.1) Get elders features and targets
elders_X, elders_Y = read_elders_cohorts(select_features=SELECTED_FEATURES)
print("Elders features shape:", elders_X.shape)

# 1.2) Get children features and targets
children_X, children_Y = read_children_cohorts(select_features=SELECTED_FEATURES)
print("Children features shape:", children_X.shape)

"""

#########
# 2) CCA: Canonical Correlation Analysis

# 2.1) Make subset for feature transformation
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
    children_Y_sub = children_Y[children_X_sub.index]

    if len(children_X_sub) < n_elders:
        # sample the elders
        elders_X_sub = elders_X_sub.sample(len(children_X_sub))
    elders_Y_sub = elders_Y[elders_X_sub.index]

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

# 2.2) Fit CCA
cca = CCA(n_components=3)
cca.fit(elders_X_trimmed, children_X_trimmed)
print("CCA fitted")


#########

# 3) Transform the whole datasets
transformed_elders_X, transformed_children_X = cca.transform(elders_X, children_X)
print("Transformed Elders features shape:", transformed_elders_X.shape)
print("Transformed Children features shape:", transformed_children_X.shape)

#########
"""

# 4.1) Model
model = GradientBoostingRegressor(n_estimators=1000, max_depth=10, random_state=0, loss='absolute_error',
                                  learning_rate=0.04,)
print(model)

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
        #assert 0 <= mmse <= 30, "MMSE must be between 0 and 30"
        #assert 0 <= age, "Developmental age estimate must be positive"

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
    plt.scatter(accurate_x, accurate_y, color="#34AC8B", marker='.', alpha=0.8)
    plt.scatter(inaccurate_x, inaccurate_y, color='red', marker='.', alpha=0.8)
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
