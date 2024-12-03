from os import makedirs
from os.path import exists

from combat_variations import *
from plotting import *
from read import *

out_path = "/Users/saraiva/PycharmProjects/LTBio/phd_journal/24_11_22/each_variation"
harmonization_method = "none"

#############

# Input-output directory
variation_out_path = join(out_path, harmonization_method)
if not exists(variation_out_path):
    raise NotADirectoryError(f"Processed data not in directory {variation_out_path} because it does not exist.")

# Read datasets
datasets = read_all_transformed_datasets(variation_out_path)
# Read metadata
metadata = read_all_metadata()

#############

# --- Draw plots and export them ---

# 1. Batch effects distribution
plot_batch_effects_dist(variation_out_path, variation_out_path)

# 2. Q-Q plots
plot_qq(datasets, variation_out_path)

# 3. Mean-diffs
plot_mean_diffs(datasets, variation_out_path)

# 4. Babiloni qualitaty control
plot_babiloni_quality_control(variation_out_path, variation_out_path)

# Concatenate datasets
datasets_concatenated = pd.concat(datasets.values(), axis=0)
metadata_concatenated = pd.concat(metadata.values(), axis=0)

# 5. tSNE and LDA
plot_2components(metadata_concatenated, variation_out_path, variation_out_path, method='tsne')
plot_2components(metadata_concatenated, variation_out_path, variation_out_path, method='lda')

# 6. Distance Matrix
plot_distance_matrix(variation_out_path, variation_out_path)

# 7. Correlation with var
plot_correlation_with_var(variation_out_path, variation_out_path, "SITE", harmonization_method.capitalize())
plot_correlation_with_var(variation_out_path, variation_out_path, "DIAGNOSIS", harmonization_method.capitalize())

# 8. Classification with var
plot_classification_with_var(variation_out_path, variation_out_path, "SITE", harmonization_method.capitalize())
plot_classification_with_var(variation_out_path, variation_out_path, "DIAGNOSIS", harmonization_method.capitalize())

