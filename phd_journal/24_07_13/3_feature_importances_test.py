import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.inspection import permutation_importance

from read import read_children, load_model, feature_names
from utils import curate_feature_names

# Get trained model
model = load_model()

# Get children data
features, targets = read_children()

# Feature importance with permutation test
X = np.array(features)
y = np.array(targets)
result = permutation_importance(model, X, y, n_repeats=8, random_state=0, n_jobs=-1)
sorted_idx = result.importances_mean.argsort()

# Get top 15 according to mean importance
importances = result.importances_mean
indices = np.argsort(importances)[::-1]
top_15_features = importances[indices[:15]]
top_15_features_names = [feature_names[i] for i in indices[:15]]
np.savetxt("./top_15_feature_importances_test.txt", top_15_features_names, fmt='%s')

# Make one color by category (Spectral, Hjorth, COH or PLI)
# original color
og_color = '#C60E4F'
# same colour but with a little less contrast
more_contrast = '#FF6699'
# same colour but with less contrast
less_contrast = '#FF99CC'

colors = []
for name in top_15_features_names:
    if 'Spectral' in name:
        colors.append(more_contrast)
    elif 'Hjorth' in name:
        colors.append(less_contrast)
    elif 'COH' or 'PLI' in name:
        colors.append(og_color)

# Curate feature names
top_15_features_names = curate_feature_names(top_15_features_names)

# Bar plot of top 15 features with seaborn
plt.rcParams['font.family'] = 'Arial'
sns.barplot(x=top_15_features, y=top_15_features_names, palette=colors)
plt.xlabel('Feature Importance', fontsize=11)
plt.yticks(fontsize=11)
plt.xticks(fontsize=11)

# remove top and right spines
sns.despine() 

plt.savefig("/Users/saraiva/Desktop/Doktorand/Scientific Outputs/Journal Articles/RH-images/" + "3_feature_importances_Bi.png", dpi=300, bbox_inches='tight')






