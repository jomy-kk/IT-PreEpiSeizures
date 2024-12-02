from os import makedirs

from combat_variations import *
from processing import *
from read import *

out_path = "/Users/saraiva/PycharmProjects/LTBio/phd_journal/24_11_22/each_variation"
harmonization_method = "none"
cov_age = True
cov_gender = True
cov_education = True
cov_diagnosis = True

#############

# Read data
datasets = read_all_datasets()
datasets_metadata = read_all_metadata()

# Discard rows that don't mean our DIAGNOSIS definitions
# Criteria to keep a row:
# AD criteria: AD & (MSE<24 | MoCA<=18)
# HC criteria: HC & (MMSE>26 | MoCA>=24)
for dataset_name, metadata in datasets_metadata.items():
    # Find rows to discard
    rows_to_discard = []
    for i, row in metadata.iterrows():  # be careful because column MoCA may not exist
        if row['DIAGNOSIS'] == 'AD' and (row['MMSE'] >= 24 or (row['MoCA'] > 18 if 'MoCA' in metadata.columns else False)):
            rows_to_discard.append(i)
        elif row['DIAGNOSIS'] == 'HC' and (row['MMSE'] <= 26 or (row['MoCA'] < 24 if 'MoCA' in metadata.columns else False)):
            rows_to_discard.append(i)
        elif row['DIAGNOSIS'] == 'MCI':
            rows_to_discard.append(i)
    # Discard rows in metadata
    datasets_metadata[dataset_name] = datasets_metadata[dataset_name].drop(rows_to_discard)
    # Discard rows in datasets
    rows_to_discard = [i for i in datasets[dataset_name].index if i not in datasets_metadata[dataset_name].index]
    datasets[dataset_name] = datasets[dataset_name].drop(rows_to_discard)
    # Ensure that there is no extra metadata
    datasets_metadata[dataset_name] = datasets_metadata[dataset_name].loc[datasets[dataset_name].index]

# Print all DIAGNOSIS values is datasets_metadata
for dataset_name, metadata in datasets_metadata.items():
    print(f"{dataset_name}: {metadata['DIAGNOSIS'].unique()}")
    print(f"{dataset_name}: Length: {len(datasets[dataset_name])}, Length metadata: {len(metadata)}")
    assert len(metadata) == len(datasets[dataset_name])


# Harmonization Step
datasets, dist_parameters = apply_combat(datasets, datasets_metadata,
                                         log_transform=True, harmonization_method=harmonization_method,
                                         cov_age=cov_age, cov_gender=cov_gender, cov_education=cov_education, cov_diagnosis=True)

# Save output datasets
variation_out_path = join(out_path, harmonization_method)
makedirs(variation_out_path, exist_ok=True)
for dataset_name, dataset in datasets.items():
    dataset.to_csv(join(variation_out_path, f"{dataset_name}.csv"))

# --- Process data for plots ---

# 1. Batch effects distribution
#batch_effects_distribution(dist_parameters, variation_out_path)

# 2. Q-Q plots
# Noting to prepare.

# 3. Mean-diffs
#mean_diffs(datasets, variation_out_path)

# 4. Babiloni qualitaty control
#babiloni_quality_control(datasets, datasets_metadata, variation_out_path)

# Concatenate datasets
datasets_concatenated = pd.concat(datasets.values(), axis=0)
datasets_metadata_concatenated = pd.concat(datasets_metadata.values(), axis=0)

# 5. tSNE and LDA
#tsne_lda(datasets_concatenated, datasets_metadata_concatenated, variation_out_path)

# 6. Distance Matrix
#distance_matrix(datasets, variation_out_path)

# 7. Correlation with var
#correlation_with_var(datasets_concatenated, datasets_metadata_concatenated, variation_out_path)

# 8. Classification with var
classification_with_var(datasets_concatenated, datasets_metadata_concatenated, variation_out_path)

