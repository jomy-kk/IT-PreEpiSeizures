import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from sklearn.inspection import plot_partial_dependence, partial_dependence

from read import feature_names, read_children, load_model
from utils import curate_feature_names, CHILDREN_COLOUR

# Get elders and children data
features_children, _ = read_children()

# Get only important features from children
important_children = np.loadtxt(f"./top_15_feature_importances_test_wr2.txt", dtype=str, comments=None)

# Get only important features from elders
important_elders = np.loadtxt(f"./top_15_feature_importances_train.txt", dtype=str, comments=None)

# Get the union of important features
important_features = np.union1d(important_children, important_elders)

# Filter
features_children = features_children[important_features]
feature_names = [name for name in feature_names if name in important_features]


# Load model
model = load_model()

# Make directory for images
dir_path = "/Users/saraiva/Desktop/Doktorand/2. Scientific Outputs/Journal Articles/RH-images/after MAS11 fixed/pdp_pairs_children"
if not os.path.exists(dir_path):
    os.makedirs(dir_path)

def custom_partial_dependence(estimator, X, features, percentiles=100):
    # Calculate the partial dependence values
    res = partial_dependence(estimator, X, features, grid_resolution=percentiles, percentiles=(0, 1))
    feature_values = res['values']
    pdp = res['average']
    return feature_values, pdp, [np.percentile(X[feature], np.arange(0, percentiles+1, 1)) for feature in features]

# define arial 12pt font for both axes ticks and y-axis label; and 16pt for x-axis label
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 10
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10

cmap = LinearSegmentedColormap.from_list("my_cmap", ["white", CHILDREN_COLOUR], N=256)


# Iterate over pairs of features
for i, feature1 in enumerate(features_children.columns):
    for j, feature2 in enumerate(features_children.columns):
        if i < j:  # To avoid duplicate pairs and self-pairs
            print(f"Processing pair {feature1} and {feature2}...")

            plt.figure(figsize=(3, 3))

            plot_partial_dependence(estimator=model,
                                    X=features_children,
                                    features=[(feature1, feature2)],
                                    feature_names=feature_names,
                                    contour_kw={'cmap': cmap, 'linewidths': 2},
                                    centered=True,
                                    n_cols=1,
                                    n_jobs=-1)

            # Labels
            plt.xlabel(curate_feature_names([feature1])[0], fontsize=14)
            plt.ylabel(curate_feature_names([feature2])[0], fontsize=12)

            name = " $ ".join(curate_feature_names([feature1, feature2]))

            # Save the figure in png with transparent background
            #sns.despine()
            plt.savefig(os.path.join(dir_path, name+'.png'), dpi=300, bbox_inches='tight', transparent=True)

            # Close the figure to free up memory
            plt.close()
