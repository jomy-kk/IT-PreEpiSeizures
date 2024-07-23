# We will compare the distribution of 6 features in two populations.
# Each population will have 3 groups.
# At the end we want violin plots with seaborn.

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from read import read_elders, feature_names, read_children
from utils import curate_feature_names

# Get elders and children data
features_elders, targets_elders = read_elders()
features_children, targets_children = read_children()


# Groups with targets boundaries
#age_classes = ((0,8), (15, 20))
age_classes = ((0,7), (7, 12), (12, 20))
#mmse_classes = ((0,15), (24, 31))
mmse_classes = ((0, 15), (16, 23), (24, 30))

# Assign groups and population
features_elders["Group"] = np.nan
features_elders["Population"] = np.nan
features_children["Group"] = np.nan
features_children["Population"] = np.nan

for i, (start, end) in enumerate(age_classes):
    features_children.loc[(targets_children >= start) & (targets_children <= end), "Group"] = i
    features_children.loc[(targets_children >= start) & (targets_children <= end), "Population"] = "Children"

for i, (start, end) in enumerate(mmse_classes):
    features_elders.loc[(targets_elders >= start) & (targets_elders <= end), "Group"] = i
    features_elders.loc[(targets_elders >= start) & (targets_elders <= end), "Population"] = "Elders"

# Drop NaN values
features_elders.dropna(inplace=True)
features_children.dropna(inplace=True)


# Features to join
#"""
FEATURES = [
    # from PDPs
    "COH#Temporal(L)-Temporal(R)#alpha",
    "COH#Frontal(L)-Temporal(R)#beta",
    "Spectral#EdgeFrequency#O2#beta",
    "COH#Frontal(R)-Parietal(L)#theta",
    "COH#Frontal(R)-Temporal(L)#theta",
    "Hjorth#Complexity#T3",

    # from children's permutation importance
    "COH#Frontal(R)-Parietal(L)#delta",
    "COH#Frontal(R)-Temporal(L)#delta",
    "COH#Temporal(R)-Parietal(R)#alpha",
    "Spectral#Entropy#T4#theta",
    "COH#Temporal(L)-Parietal(R)#delta",

    # from elders' decrease in impurity importance
    "Spectral#Flatness#P3#beta",
    "Spectral#RelativePower#F7#theta",
    "Spectral#Entropy#O2#alpha",
    "COH#Frontal(R)-Parietal(L)#gamma",
    "Hjorth#Mobility#P4",
    "COH#Temporal(L)-Parietal(L)#gamma",
]
#"""
#FEATURES = feature_names

##################
# PCA
from sklearn.decomposition import PCA

# Join features
features_elders.reset_index(drop=True, inplace=True)
features_children.reset_index(drop=True, inplace=True)
features = pd.concat([features_elders, features_children])

# Remove "Group" and "Population" columns
group = features["Group"]
population = features["Population"]
features = features.drop(columns=["Group", "Population"])

# Select important features
features = features.loc[:, FEATURES]

# Normalize features between 0 and 1
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
features = scaler.fit_transform(features)


"""
# Apply PCA
pca = PCA(n_components=3)
features_pca = pca.fit_transform(features)
"""

# Apply t-SNE
"""
from sklearn.manifold import TSNE
tsne = TSNE(n_components=3, perplexity=10, n_iter=3000,
            random_state=42)
features_pca = tsne.fit_transform(features)
"""

# Apply UMAP
#"""
from umap import UMAP
umap = UMAP(n_components=2, n_neighbors=10, min_dist=0.1)
features_pca = umap.fit_transform(features)
#"""

# Add "Group" and "Population" columns
features_pca = pd.DataFrame(features_pca, columns=["PC1", "PC2"])
group.reset_index(drop=True, inplace=True)
features_pca["Group"] = group
population.reset_index(drop=True, inplace=True)
features_pca["Population"] = population
markers = {"Children": "s", "Elders": "o"}  # triangles for children and circles for elders

# Plot 2D PCA
#"""
plt.figure(figsize=(10, 10))
sns.scatterplot(x="PC1", y="PC2", hue="Group", style="Population", markers=markers, data=features_pca, alpha=0.7)
plt.show()
#"""

# Plot 3D PCA
"""
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection='3d')
ax.set_title("PCA")

# make 3 colors, one for each group
colours = ['r', 'g', 'b']

for g, group in enumerate(features_pca["Group"].unique()):
    for population in features_pca["Population"].unique():
        mask = (features_pca["Group"] == group) & (features_pca["Population"] == population)
        ax.scatter(features_pca.loc[mask, "PC1"], features_pca.loc[mask, "PC2"], features_pca.loc[mask, "PC3"], label=f"{group} - {population}", marker=markers[population], color=colours[g])
ax.set_xlabel("PC1")
ax.set_ylabel("PC2")
ax.set_zlabel("PC3")
plt.legend()
plt.show()
"""

