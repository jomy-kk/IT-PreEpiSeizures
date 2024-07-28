from pickle import load

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from utils import curate_feature_names

# Get trained model
model = load(open('/Users/saraiva/PycharmProjects/LTBio/phd_journal/24_03_18/inverse_problem3/scheme57/model.pkl', 'rb'))

# Feature names
feature_names = ['Hjorth#Complexity#T5', 'Hjorth#Complexity#F4',
       'COH#Frontal(R)-Parietal(L)#delta', 'Hjorth#Complexity#T3',
       'Spectral#RelativePower#F7#theta', 'COH#Frontal(R)-Temporal(L)#theta',
       'Spectral#EdgeFrequency#O2#beta', 'COH#Frontal(L)-Temporal(R)#beta',
       'COH#Temporal(L)-Parietal(L)#gamma', 'Spectral#EdgeFrequency#O1#beta',
       'COH#Frontal(R)-Parietal(L)#theta', 'COH#Temporal(L)-Temporal(R)#alpha',
       'COH#Frontal(R)-Temporal(L)#gamma', 'COH#Temporal(R)-Parietal(L)#beta',
       'COH#Frontal(R)-Occipital(L)#theta', 'COH#Temporal(L)-Parietal(L)#beta',
       'Hjorth#Activity#F7', 'COH#Occipital(L)-Occipital(R)#gamma',
       'Spectral#Flatness#P3#beta', 'COH#Temporal(R)-Parietal(R)#alpha',
       'Spectral#Entropy#P3#alpha', 'COH#Frontal(R)-Parietal(R)#theta',
       'COH#Frontal(R)-Temporal(L)#delta', 'Spectral#Entropy#O2#alpha',
       'Spectral#Entropy#T4#theta', 'Spectral#RelativePower#Cz#beta',
       'Spectral#Diff#Pz#delta', 'COH#Parietal(R)-Occipital(L)#beta',
       'Spectral#EdgeFrequency#Fz#beta', 'Spectral#Diff#Cz#gamma',
       'Spectral#RelativePower#Fp1#gamma', 'COH#Frontal(R)-Parietal(L)#gamma',
       'PLI#Frontal(R)-Parietal(L)#alpha', 'Spectral#Diff#F7#beta',
       'Hjorth#Mobility#O1', 'Spectral#Flatness#T4#gamma',
       'PLI#Parietal(L)-Occipital(L)#gamma', 'Spectral#Flatness#T6#delta',
       'COH#Parietal(R)-Occipital(L)#alpha',
       'COH#Parietal(R)-Occipital(R)#beta', 'Spectral#Diff#T4#delta',
       'Spectral#Diff#F8#alpha', 'COH#Temporal(R)-Occipital(L)#beta',
       'COH#Parietal(R)-Occipital(L)#gamma', 'Hjorth#Mobility#P4',
       'COH#Frontal(L)-Temporal(L)#beta',
       'COH#Occipital(L)-Occipital(R)#alpha', 'Spectral#Entropy#T3#theta',
       'COH#Frontal(R)-Occipital(R)#alpha', 'Hjorth#Complexity#P3',
       'COH#Frontal(L)-Occipital(L)#beta', 'Hjorth#Activity#C3',
       'COH#Temporal(L)-Occipital(R)#theta', 'Spectral#Diff#F4#beta',
       'COH#Frontal(L)-Frontal(R)#gamma', 'Spectral#Diff#C3#gamma',
       'COH#Frontal(L)-Frontal(R)#theta', 'COH#Parietal(L)-Occipital(R)#theta',
       'Spectral#RelativePower#F7#gamma', 'Spectral#RelativePower#F3#beta',
       'PLI#Temporal(R)-Parietal(R)#beta', 'Spectral#Flatness#F7#beta',
       'Hjorth#Complexity#O2', 'Spectral#Entropy#Cz#theta',
       'PLI#Frontal(R)-Occipital(R)#beta', 'COH#Temporal(L)-Parietal(R)#beta',
       'COH#Frontal(L)-Occipital(L)#delta', 'Spectral#Flatness#F8#delta',
       'Spectral#Entropy#F4#delta', 'PLI#Temporal(R)-Parietal(R)#gamma',
       'COH#Occipital(L)-Occipital(R)#delta',
       'COH#Temporal(L)-Parietal(R)#delta', 'PLI#Frontal(L)-Temporal(R)#delta',
       'Spectral#Flatness#P3#theta', 'Spectral#Entropy#F7#alpha',
       'COH#Frontal(R)-Temporal(R)#delta', 'COH#Frontal(L)-Occipital(R)#gamma',
       'COH#Frontal(L)-Frontal(R)#beta', 'Hjorth#Complexity#Cz',
       'COH#Frontal(L)-Occipital(R)#beta']

# Get feature importances
importances = model.feature_importances_

# Sort feature importances
indices = np.argsort(importances)[::-1]

# Select top 15 features
top_15_features = importances[indices[:15]]
top_15_features_names = [feature_names[i] for i in indices[:15]]
np.savetxt("./top_15_feature_importances_train.txt", top_15_features_names, fmt='%s')

# Curate feature names
top_15_features_names = curate_feature_names(top_15_features_names)

# Make colors
in_common_color = '#C60E4F'
lightgray_color = '#D3D3D3'

in_common_features = (
    "Hjorth Complexity T3",
    "Edge Frequency O2 Beta",
    "COH Frontal(R) - Parietal(L) Theta",
    "COH Temporal(L) - Temporal(R) Alpha",
    "COH Frontal(R) - Temporal(L) Theta",
    "Entropy O2 Alpha",
    "COH Frontal(R) - Parietal(L) Gamma",
    "Hjorth Mobility P4"
)

colors = []
for name in top_15_features_names:
    if name in in_common_features:
        colors.append(in_common_color)
    else:
        colors.append(lightgray_color)

# Bar plot of top 15 features with seaborn
plt.rcParams['font.family'] = 'Arial'
sns.barplot(x=top_15_features, y=top_15_features_names, palette=colors)
plt.xlabel('Feature Importance', fontsize=11)
plt.yticks(fontsize=11)
plt.xticks(fontsize=11)

# remove top and right spines
sns.despine()

plt.savefig("/Users/saraiva/Desktop/Doktorand/Scientific Outputs/Journal Articles/RH-images/" + "1_feature_importances_Ai.png", dpi=300, bbox_inches='tight')






