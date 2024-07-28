import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.inspection import permutation_importance, plot_partial_dependence, partial_dependence
from sklearn.metrics.pairwise import cosine_similarity

from read import read_elders, feature_names, read_children, load_model
from utils import curate_feature_names, CHILDREN_COLOUR

# Get elders and children data
#features_elders, targets_elders = read_elders()
features_children, targets_children = read_children()

# Load model
model = load_model()

# Make directory for images
dir_path = "/Users/saraiva/Desktop/Doktorand/Scientific Outputs/Journal Articles/RH-images/pdp_children"
if not os.path.exists(dir_path):
    os.makedirs(dir_path)

#feature_names = curate_feature_names(list(features_children.columns))


def custom_partial_dependence(estimator, X, feature, percentiles=100):
    # Calculate the partial dependence values
    res = partial_dependence(estimator, X, [feature], grid_resolution=percentiles, percentiles=(0, 1))
    feature_values = res['values']
    pdp = res['average']
    return feature_values, pdp, np.percentile(X[feature], np.arange(0, percentiles+1, 1))

# Usage
# custom_partial_dependence(model, features_children, 'feature_name')

# Iterate over the features
for i, feature in enumerate(features_children.columns):
    # define arial 12pt font for both axes ticks and y-axis label; and 16pt for x-axis label
    plt.rcParams['font.family'] = 'Arial'
    plt.rcParams['font.size'] = 12
    plt.rcParams['xtick.labelsize'] = 12
    plt.rcParams['ytick.labelsize'] = 12

    # Create a new figure for each feature
    plt.figure(figsize=(4, 3))

    """
    # Create a partial dependence plot for the feature
    pdp = plot_partial_dependence(estimator=model,
                                  X=features_children,
                                  features=[feature],
                                  feature_names=feature_names,
                                  line_kw={'color': CHILDREN_COLOUR, 'linewidth': 2},
                                  centered=True,
                                  n_cols=1,)
    """
    feature_values, pdp, percentiles = custom_partial_dependence(model, features_children, feature)
    feature_values = feature_values[0]
    pdp = pdp[0,:]
    plt.plot(feature_values, pdp, color=CHILDREN_COLOUR, linewidth=3)
    for p in percentiles:
        plt.vlines(p, -1.6, -1.4, colors='black', linewidth=0.5)

    plt.xlabel(curate_feature_names([feature])[0], fontsize=14)
    plt.ylabel('Partial dependence', fontsize=12)

    plt.ylim(-1.6, 3.1)
    plt.yticks((-1.5, 0, 1.5, 3), ('-1.5', '0.0', '+1.5', '+3.0'))

    # Save the figure in png with transparent background
    sns.despine()
    plt.savefig(os.path.join(dir_path, f'{feature}.png'), dpi=300, bbox_inches='tight', transparent=True)

    # Close the figure to free up memory
    plt.close()
