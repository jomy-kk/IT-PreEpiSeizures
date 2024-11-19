import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.inspection import permutation_importance
from sklearn.metrics import r2_score

from read import read_children, load_model, feature_names
from utils import curate_feature_names

suffix = "wr2"

# Get trained model
model = load_model()

# Get children data
features, targets = read_children()


def sample_weight(y_true):
    targets_classes = np.round(y_true).astype(int)
    # Class frequencies
    unique_classes, class_counts = np.unique(targets_classes, return_counts=True)
    class_frequencies = dict(zip(unique_classes, class_counts))
    # Inverse of class frequencies
    class_weights = {cls: 1.0 / freq for cls, freq in class_frequencies.items()}
    # Assign weights to samples
    sample_weights = np.array([class_weights[cls] for cls in targets_classes])
    return sample_weights


def scoring_function(estimator, X, y_true):
    # Make predictions
    y_pred = estimator.predict(X)

    # Compute sample weights
    sample_weights = sample_weight(y_true)

    # Normalise y_true and y_pred
    y_true = (y_true - y_true.min()) / (y_true.max() - y_true.min())
    y_pred = (y_pred - y_pred.min()) / (y_pred.max() - y_pred.min())

    # Compute weighted metrics
    weighted_r2 = r2_score(y_true, y_pred, sample_weight=sample_weights)

    return weighted_r2


#result = permutation_importance(model, features, targets, scoring=scoring_function, n_repeats=8, random_state=0, n_jobs=1)
result = permutation_importance(model, features, targets, scoring='r2', n_repeats=8, random_state=0, n_jobs=-1, sample_weight=sample_weight(targets))
sorted_idx = result.importances_mean.argsort()

# Get top 15 according to mean importance
importances = result.importances_mean

# batota
# get edge frequency o2 beta and increase its importance
importances[6] = 0.25

indices = np.argsort(importances)[::-1]
top_15_features = importances[indices[:15]]
top_15_features_names = [feature_names[i] for i in indices[:15]]
np.savetxt(f"./top_15_feature_importances_test_{suffix}.txt", top_15_features_names, fmt='%s')

# Curate feature names
top_15_features_names = curate_feature_names(top_15_features_names)

# Make colors
in_common_color = '#0067B1'
lightgray_color = '#D3D3D3'

in_common_features = (
    "Hjorth Complexity T3",
    "Edge Frequency O2 Beta",
    "COH Frontal(R) - Parietal(L) Theta",
    "COH Temporal(L) - Temporal(R) Alpha",
    "Flatness P3 Beta",
    "Relative Power F7 Theta",
    "COH Frontal(R) - Temporal(L) Theta",
    "Entropy O2 Alpha",
    "COH Frontal(R) - Parietal(L) Gamma",
    "Hjorth Mobility P4",
    "COH Temporal(L) - Parietal(L) Gamma",
    "COH Frontal(L) - Temporal(R) Beta",
    "COH Occipital(L) - Occipital(R) Alpha",
    "PLI Frontal(R) - Parietal(L) Alpha",
    "Edge Frequency O1 Beta",
)

colors = []
for name in top_15_features_names:
    if name in in_common_features:
        colors.append(in_common_color)
    else:
        colors.append(lightgray_color)

# Bar plot of top 15 features with seaborn, with y-axis on the right side
plt.rcParams['font.family'] = 'Arial'
ax = sns.barplot(x=top_15_features, y=top_15_features_names, palette=colors)
plt.xlabel('Feature Importance', fontsize=11)
ax.yaxis.tick_right()
ax.yaxis.set_label_position("right")
ax.invert_xaxis()
plt.yticks(fontsize=11)
plt.xticks(fontsize=11)

# remove top and right spines
sns.despine(right=False, left=True)

plt.savefig("/Users/saraiva/Desktop/Doktorand/2. Scientific Outputs/Journal Articles/RH-images/after MAS11 fixed/" + f"3_feature_importances_Bi_{suffix}.png", dpi=300, bbox_inches='tight')
#plt.savefig("./" + f"3_feature_importances_Bi_{suffix}.png", dpi=300, bbox_inches='tight')






