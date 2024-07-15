import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.inspection import permutation_importance, plot_partial_dependence
from sklearn.metrics.pairwise import cosine_similarity

from read import read_elders, feature_names, read_children, load_model
from utils import curate_feature_names

# Get elders and children data
#features_elders, targets_elders = read_elders()
features_children, targets_children = read_children()

# Load model
model = load_model()

# Make directory for images
dir_path = "/Users/saraiva/Desktop/Doktorand/Scientific Outputs/Journal Articles/RH-images/pdp_children"
if not os.path.exists(dir_path):
    os.makedirs(dir_path)

feature_names = curate_feature_names(list(features_children.columns))

# Create partial dependence plots
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['axes.labelsize'] = 16

"""
plt.figure(figsize=(30, 30))
pdp = plot_partial_dependence(estimator=model,
                        X=features_children,
                        features=feature_names,
                        feature_names=feature_names,
                        n_cols=6,
                        n_jobs=-1)

fig = pdp.figure_
ax = pdp.axes_

sns.despine()

# Adjust the layout of the subplots
plt.subplots_adjust(wspace=0.4, hspace=0.6)

fig.savefig(dir_path + '.png', dpi=300, bbox_inches='tight')
"""

# Iterate over the features
for i, feature in enumerate(feature_names):
    # Create a new figure for each feature
    plt.figure(figsize=(3, 3))

    # Create a partial dependence plot for the feature
    pdp = plot_partial_dependence(estimator=model,
                                  X=features_children,
                                  features=[feature],
                                  feature_names=feature_names,
                                  centered=True,
                                  n_cols=1,)

    plt.ylim(-1.5, 3.1)

    # Save the figure in png with transparent background
    sns.despine()
    plt.savefig(os.path.join(dir_path, f'{feature}.png'), dpi=300, bbox_inches='tight', transparent=True)

    # Close the figure to free up memory
    plt.close()