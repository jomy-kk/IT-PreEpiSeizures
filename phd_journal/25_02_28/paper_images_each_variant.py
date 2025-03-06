from os.path import exists

from plotting import *
from read import *


#############

def make_images(run, out_path, dataset_names, variant, model):

    # Input-output directory
    variation_out_path = variant
    out_path = join(out_path, str(model))
    if not exists(out_path):
        raise NotADirectoryError(f"Processed data not in directory {variation_out_path} because it does not exist.")

    # Read metadata
    metadata = read_all_metadata(dataset_names)
    metadata_concatenated = pd.concat(metadata.values(), axis=0)

    plot_simple_diagnosis_discriminant(run, metadata_concatenated, out_path, out_path)

    plot_classification_with_var(run, out_path, out_path, "DIAGNOSIS", str(model))
