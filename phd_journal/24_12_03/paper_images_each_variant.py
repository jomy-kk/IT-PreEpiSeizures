from os.path import exists

from plotting import *
from read import *


#############

def make_images(run, out_path, dataset_names, harmonization_method):

    # Input-output directory
    variation_out_path = join(out_path, harmonization_method)
    if not exists(variation_out_path):
        raise NotADirectoryError(f"Processed data not in directory {variation_out_path} because it does not exist.")

    # Read datasets
    datasets = read_all_transformed_datasets(variation_out_path, dataset_names)
    # Read metadata
    metadata = read_all_metadata(dataset_names)

    #############

    # --- Draw plots and export them ---

    # 0. MMSE distribution
    #plot_mmse_distribution(run, metadata, variation_out_path, variation_out_path)

    # 1. Batch effects distribution
    #if harmonization_method != "none" and harmonization_method != "original":
    #    plot_batch_effects_dist(run, variation_out_path, variation_out_path)

    # 2. Q-Q plots
    #plot_qq(run, datasets, variation_out_path)

    # 3. Mean-diffs
    plot_mean_diffs(run, datasets, variation_out_path)

    # 4. Babiloni qualitaty control
    plot_babiloni_quality_control(run, variation_out_path, variation_out_path)

    # Concatenate datasets
    #datasets_concatenated = pd.concat(datasets.values(), axis=0)
    metadata_concatenated = pd.concat(metadata.values(), axis=0)

    # 5. tSNE and LDA
    #plot_2components(run, metadata_concatenated, variation_out_path, variation_out_path, method='tsne')
    plot_2components(run, metadata_concatenated, variation_out_path, variation_out_path, method='lda')

    # 6. Distance Matrix
    plot_distance_matrix(run, variation_out_path, variation_out_path)

    # 7. Correlation with var
    plot_correlation_with_var(run, variation_out_path, variation_out_path, "SITE", harmonization_method.capitalize())
    plot_correlation_with_var(run, variation_out_path, variation_out_path, "DIAGNOSIS", harmonization_method.capitalize())

    plot_simple_diagnosis_discriminant(run, metadata_concatenated, variation_out_path, variation_out_path)

    # 8. Classification with var
    #plot_classification_with_var(run, variation_out_path, variation_out_path, "SITE", harmonization_method.capitalize())
    plot_classification_with_var(run, variation_out_path, variation_out_path, "DIAGNOSIS", harmonization_method.capitalize())
