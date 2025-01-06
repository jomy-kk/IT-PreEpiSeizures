from os import makedirs

from combat_variations import *
from processing import *
from read import *

out_path = "/Users/saraiva/PycharmProjects/LTBio/phd_journal/24_11_22/each_variation"
harmonization_method = "original"
cov_age = True
cov_gender = True
cov_education = False
cov_diagnosis = True

MMSE_criteria = (23, 27)
MoCA_criteria = (18, 24)

#############

variation_out_path = join(out_path, harmonization_method)
makedirs(variation_out_path, exist_ok=True)

# Read data
datasets = read_all_datasets(['Izmir', 'Newcastle', 'Miltiadous', 'Istambul', 'BrainLat:CL', 'BrainLat:AR'])
datasets_metadata = read_all_metadata(['Izmir', 'Newcastle', 'Miltiadous', 'Istambul', 'BrainLat:CL', 'BrainLat:AR'])

# Discard rows that don't mean our DIAGNOSIS definitions
# Criteria to keep a row:
# AD criteria: AD & (MSE<24 | MoCA<=18)
# HC criteria: HC & (MMSE>26 | MoCA>=24)
for dataset_name, metadata in datasets_metadata.items():
    # Find rows to discard
    rows_to_keep = []
    for i, row in metadata.iterrows():
        if row['DIAGNOSIS'] == 'AD':
            if 'MoCA' in metadata.columns and pd.notna(row['MoCA']) and row['MoCA'] != ' ' and int(row['MoCA']) <= MoCA_criteria[0]:
                rows_to_keep.append(i)
            elif 'MMSE' in metadata.columns and pd.notna(row['MMSE']) and row['MMSE'] != ' ' and int(row['MMSE']) <= MMSE_criteria[0]:
                rows_to_keep.append(i)
        elif row['DIAGNOSIS'] == 'HC':
            if 'MoCA' in metadata.columns and pd.notna(row['MoCA']) and row['MoCA'] != ' 'and int(row['MoCA']) >= MoCA_criteria[1]:
                rows_to_keep.append(i)
            elif 'MMSE' in metadata.columns and pd.notna(row['MMSE']) and row['MMSE'] != ' ' and int(row['MMSE']) >= MMSE_criteria[1]:
                rows_to_keep.append(i)
    # Discard rows in metadata
    datasets_metadata[dataset_name] = datasets_metadata[dataset_name].loc[rows_to_keep]
    # Discard rows in datasets
    rows_to_discard = [i for i in datasets[dataset_name].index if i not in datasets_metadata[dataset_name].index]
    datasets[dataset_name] = datasets[dataset_name].drop(rows_to_discard)
    # Ensure that there is no extra metadata
    datasets_metadata[dataset_name] = datasets_metadata[dataset_name].loc[datasets[dataset_name].index]

# Print all DIAGNOSIS values is datasets_metadata
"""
for dataset_name, metadata in datasets_metadata.items():
    print(f"\nDataset: {dataset_name}")
    print(f"Total: {len(metadata)}")
    for diagnosis in metadata['DIAGNOSIS'].unique():
        print(f"{diagnosis}: {len(metadata[metadata['DIAGNOSIS'] == diagnosis])}")
"""

# Harmonization Step
datasets, dist_parameters = apply_combat(datasets, datasets_metadata,
                                         log_transform=True, harmonization_method=harmonization_method,
                                         cov_age=cov_age, cov_gender=cov_gender, cov_education=cov_education, cov_diagnosis=cov_diagnosis)

for dataset_name, _ in datasets.items():
    out_filepath = join(variation_out_path, f"{dataset_name}.csv")
    datasets[dataset_name].to_csv(out_filepath)


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

pc = simple_diagnosis_discriminant(datasets_concatenated, datasets_metadata_concatenated, variation_out_path,
                                  method='pca', norm_method='none', n_components=11,
                                  relevant_features='all'
                                  #relevant_features=['Alpha1_Frontal', 'Theta_Temporal', 'Alpha3_Occipital']
                                  )

# 8. Classification with var
classification_with_var(datasets_concatenated, datasets_metadata_concatenated, variation_out_path, "DIAGNOSIS")

