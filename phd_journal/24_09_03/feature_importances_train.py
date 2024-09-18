from pickle import load

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from utils import curate_feature_names

# Get trained model
model = load(open('/Users/saraiva/PycharmProjects/LTBio/phd_journal/24_09_03/scheme10/model.pkl', 'rb'))

# Feature names
feature_names = ['COH#Frontal(R)-Temporal(L)#alpha', 'Hjorth#Mobility#F4',
                                      'Spectral#RelativePower#C4#delta', 'Spectral#Flatness#Pz#gamma',
                                      'Spectral#Flatness#P3#gamma', 'Spectral#RelativePower#Fpz#delta',
                                      'Spectral#RelativePower#C3#beta', 'Spectral#Flatness#T4#beta',
                                      'Spectral#RelativePower#C4#beta', 'COH#Frontal(R)-Occipital(R)#alpha',
                                      'Spectral#RelativePower#Fz#beta', 'Spectral#RelativePower#Cz#beta',
                                      'COH#Frontal(L)-Occipital(L)#beta', 'Hjorth#Mobility#Fz',
                                      'Spectral#Diff#F7#gamma', 'Spectral#RelativePower#C3#delta', 'Hjorth#Mobility#C3',
                                      'Spectral#RelativePower#Fz#delta', 'Hjorth#Mobility#C4',
                                      'Spectral#EdgeFrequency#P3#alpha', 'Spectral#Flatness#P4#gamma',
                                      'Hjorth#Complexity#C4', 'Spectral#EdgeFrequency#F8#gamma',
                                      'COH#Frontal(R)-Parietal(L)#theta', 'Spectral#EdgeFrequency#T4#alpha',
                                      'COH#Frontal(L)-Parietal(R)#gamma', 'Spectral#RelativePower#Cz#gamma',
                                      'Hjorth#Mobility#Cz', 'COH#Frontal(L)-Parietal(R)#alpha',
                                      'COH#Temporal(R)-Parietal(R)#theta']

# Get feature importances
importances = model.feature_importances_

# Sort feature importances
indices = np.argsort(importances)[::-1]

# Select top 10 features
top_10_features = importances[indices[:15]]
top_10_features_names = [feature_names[i] for i in indices[:15]]
np.savetxt("./top_15_feature_importances_train.txt", top_10_features_names, fmt='%s')

# Curate feature names
top_10_features_names = curate_feature_names(top_10_features_names)

# Bar plot of top 15 features with seaborn
plt.rcParams['font.family'] = 'Arial'
sns.barplot(x=top_10_features, y=top_10_features_names, palette='#C60E4F')
plt.xlabel('Feature Importance', fontsize=11)
plt.yticks(fontsize=11)
plt.xticks(fontsize=11)

# remove top and right spines
sns.despine()

plt.savefig('/Users/saraiva/PycharmProjects/LTBio/phd_journal/24_09_03/' + "feature_importances_train.png", dpi=300, bbox_inches='tight')






