import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.inspection import permutation_importance
from sklearn.metrics.pairwise import cosine_similarity

from read import read_elders, feature_names, read_children
from utils import curate_feature_names

# Get elders and children data
features_elders, targets_elders = read_elders()
features_children, targets_children = read_children()

# Get top importances for each dataset
elders_important_features = np.loadtxt('./top_15_feature_importances_train.txt', dtype=str, comments=None)
children_important_features = np.loadtxt('./top_15_feature_importances_test_wr2.txt', dtype=str, comments=None)

# Select only important features
features_elders_Ai = features_elders.loc[:, elders_important_features]
features_children_Ai = features_children.loc[:, elders_important_features]
features_elders_Bi = features_elders.loc[:, children_important_features]
features_children_Bi = features_children.loc[:, children_important_features]

# Correlation matrices
# Ai
features_elders_Ai['Target'] = targets_elders
features_elders_Ai_corr = features_elders_Ai.corr()

features_children_Ai['Target'] = targets_children
features_children_Ai_corr = features_children_Ai.corr()

# Make one matrix using lower triangle of 'features_elders_Ai_corr' and upper triangle of 'features_children_Ai_corr'
lower_triangle_elders = np.tril(features_elders_Ai_corr, k=-1)
upper_triangle_children = np.triu(features_children_Ai_corr, k=1)
lower_triangle_children = np.tril(features_children_Ai_corr, k=-1)
features_Ai_corr = lower_triangle_elders + upper_triangle_children

# compute similarity between upper and lower triangle
similarity_ai = cosine_similarity([lower_triangle_children.flatten()], [lower_triangle_elders.flatten()])
print("Similarity Ai:", similarity_ai)

# make diagonal 1
np.fill_diagonal(features_Ai_corr, np.nan)


# Bi
features_elders_Bi['Target'] = targets_elders
features_elders_Bi_corr = features_elders_Bi.corr()

features_children_Bi['Target'] = targets_children
features_children_Bi_corr = features_children_Bi.corr()

# Make one matrix using lower triangle of 'features_elders_Bi_corr' and upper triangle of 'features_children_Bi_corr'
lower_triangle_elders = np.tril(features_elders_Bi_corr, k=-1)
upper_triangle_children = np.triu(features_children_Bi_corr, k=1)
lower_triangle_children = np.tril(features_children_Bi_corr, k=-1)
features_Bi_corr = lower_triangle_elders + upper_triangle_children

# compute similarity between upper and lower triangle
similarity_bi = cosine_similarity([lower_triangle_children.flatten()], [lower_triangle_elders.flatten()])
print("Similarity Bi:", similarity_bi)

# make diagonal 1
np.fill_diagonal(features_Bi_corr, np.nan)


# Plot heatmap
#plt.figure(figsize=(10, 10))
sns.heatmap(features_Ai_corr, cmap='coolwarm', center=0, square=True, vmin=-1, vmax=1)
labels = curate_feature_names(list(elders_important_features))
plt.xticks(np.array(np.arange(len(labels)+1)) + 0.5, labels + ['Age', ], rotation=90)
plt.yticks(np.array(np.arange(len(labels)+1)) + 0.5, labels + ['MMSE', ], rotation=0)
plt.savefig("/Users/saraiva/Desktop/Doktorand/Scientific Outputs/Journal Articles/RH-images/" + "2_feature_correlations_Ai.png", dpi=300, bbox_inches='tight')
plt.clf()

#plt.figure(figsize=(10, 10))
sns.heatmap(features_Bi_corr, cmap='coolwarm', center=0, square=True, vmin=-1, vmax=1)
labels = curate_feature_names(list(children_important_features))
plt.xticks(np.array(np.arange(len(labels)+1)) + 0.5, labels + ['Age', ], rotation=90)
plt.yticks(np.array(np.arange(len(labels)+1)) + 0.5, labels + ['MMSE', ], rotation=0)
plt.savefig("/Users/saraiva/Desktop/Doktorand/Scientific Outputs/Journal Articles/RH-images/" + "4_feature_correlations_Bi_wr2.png", dpi=300, bbox_inches='tight')
plt.clf()

