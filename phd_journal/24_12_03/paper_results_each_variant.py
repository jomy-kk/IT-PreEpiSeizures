from os import makedirs

import pandas as pd
from neptune.types import File
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import ElasticNet
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
 
from combat_variations import *
from processing import *
from read import *

#############

def make_results(run, out_path, dataset_names, harmonization_method, cov_age, cov_gender, cov_education, cov_diagnosis, MMSE_criteria, MoCA_criteria, n_pc):

    variation_out_path = join(out_path, harmonization_method)
    makedirs(variation_out_path, exist_ok=True)

    # Read data
    datasets = read_all_datasets(dataset_names)
    datasets_metadata = read_all_metadata(dataset_names)

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

    run["datasets/criteria"] = f"AD criteria: AD & (MMSE<={MMSE_criteria[0]} | MoCA<={MoCA_criteria[0]})\n" \
                                f"HC criteria: HC & (MMSE>={MMSE_criteria[1]} | MoCA>={MoCA_criteria[1]})"

    # Print all DIAGNOSIS values is datasets_metadata
    """
    for dataset_name, metadata in datasets_metadata.items():
        print(f"\nDataset: {dataset_name}")
        print(f"Total: {len(metadata)}")
        for diagnosis in metadata['DIAGNOSIS'].unique():
            print(f"{diagnosis}: {len(metadata[metadata['DIAGNOSIS'] == diagnosis])}")
    """
    for dataset_name, dataset in datasets.items():
        run[f"datasets/before/{dataset_name}"].upload(File.as_html(dataset))

    # Harmonization Step
    # Does harmonized data already exist?
    log_transform = True
    standardize = "none"
    if exists(variation_out_path) and all([exists(join(variation_out_path, f"{dataset_name}.csv")) for dataset_name in datasets.keys()]):
        print("Utilizing harmonized serialised data")
        for dataset_name in datasets.keys():
            harmonized_data_path = join(variation_out_path, f"{dataset_name}.csv")
            datasets[dataset_name] = pd.read_csv(harmonized_data_path, index_col=0)
            run[f"datasets/after/{dataset_name}_track"].track_files(harmonized_data_path)
            run[f"datasets/after/{dataset_name}"].upload(harmonized_data_path)

    else:  # No => apply harmonization
        datasets, dist_parameters = apply_combat(run, datasets, datasets_metadata, _out_path=variation_out_path,
                                                 log_transform=log_transform, standardize=standardize, harmonization_method=harmonization_method,
                                                 cov_age=cov_age, cov_gender=cov_gender, cov_education=cov_education, cov_diagnosis=cov_diagnosis)

        for dataset_name, _ in datasets.items():
            out_filepath = join(variation_out_path, f"{dataset_name}.csv")
            datasets[dataset_name].to_csv(out_filepath)
            run[f"datasets/after/{dataset_name}_track"].track_files(out_filepath)
            run[f"datasets/after/{dataset_name}"].upload(out_filepath)

    run["harmonization/params"] = {
        "method": harmonization_method if harmonization_method != "neuroharmonize" else "neuroharmonize2",  # To fix for the fact that in Neptune, 'neuroharmonize' is in fact 'neurocombat' and 'neuroharmonize2' is the true 'neuroharmonize'
        "log_transform": log_transform,
        "standardize": standardize,
        "cov_age": cov_age,
        "cov_gender": cov_gender,
        "cov_education": cov_education,
        "cov_diagnosis": cov_diagnosis
    }

    # --- Process data for plots ---

    # 1. Batch effects distribution
    #if dist_parameters is not None:
    #    batch_effects_distribution(run, dist_parameters, variation_out_path)

    # 2. Q-Q plots
    # Noting to prepare.

    # 3. Mean-diffs
    mean_diffs(run, datasets, variation_out_path)

    # 4. Babiloni qualitaty control
    babiloni_quality_control(run, datasets, datasets_metadata, variation_out_path)

    # Concatenate datasets
    datasets_concatenated = pd.concat(datasets.values(), axis=0)
    datasets_metadata_concatenated = pd.concat(datasets_metadata.values(), axis=0)

    # 5. tSNE and LDA
    tsne_lda(run, datasets_concatenated, datasets_metadata_concatenated, variation_out_path)

    # 6. Distance Matrix
    distance_matrix(run, datasets, variation_out_path)

    # 7. Correlation with var
    correlation_with_var(run, datasets_concatenated, datasets_metadata_concatenated, variation_out_path)

    # 8. Classification with var
    """
    classification_with_var(run, datasets_concatenated, datasets_metadata_concatenated, _out_path=variation_out_path,
                            var_name="DIAGNOSIS",
                            model=RandomForestClassifier(n_estimators=200, max_depth=15, random_state=0),
                            #model=SVC(kernel='linear', C=3),
                            #model=MLPClassifier(hidden_layer_sizes=(100, 50, 25, 10), alpha=0.0001, max_iter=1000, random_state=0),
                            #relevant_features=['Delta_Temporal', 'Delta_Limbic', 'Alpha1_Frontal', 'Alpha1_Limbic', 'Theta_Temporal', 'Theta_Limbic', 'Alpha3_Parietal', 'Alpha3_Occipital', 'Beta1_Parietal', 'Beta1_Occipital', 'Theta_Frontal', 'Alpha3_Temporal'],
                            relevant_features=10,
                            norm_method='min-max')
                            #)

    classification_with_var(run, datasets_concatenated, datasets_metadata_concatenated, _out_path=variation_out_path,
                            var_name="SITE",
                            model=RandomForestClassifier(n_estimators=200, max_depth=15, random_state=0),
                            relevant_features=10,
                            norm_method='min-max')
    """
    #"""
    pc = simple_diagnosis_discriminant(run, datasets_concatenated, datasets_metadata_concatenated, variation_out_path,
                                  method='pca', norm_method='none', n_components=n_pc,
                                  relevant_features='all'
                                  #relevant_features=['Alpha1_Frontal', 'Theta_Temporal', 'Alpha3_Occipital']
                                  )
    #"""
    classification_with_var(run, pc, datasets_metadata_concatenated, _out_path=variation_out_path,
                            var_name="DIAGNOSIS",
                            model=RandomForestClassifier(n_estimators=200, max_depth=15, random_state=0),
                            #model=GradientBoostingClassifier(n_estimators=200, max_depth=15, random_state=0),
                            #model=SVC(kernel='rbf', C=3),
                            #model=MLPClassifier(hidden_layer_sizes=(20, 10, 5,), max_iter=5000, random_state=0, alpha=0.0001),
                            #relevant_features=['Delta_Temporal', 'Delta_Limbic', 'Alpha1_Frontal', 'Alpha1_Limbic', 'Theta_Temporal', 'Theta_Limbic', 'Alpha3_Parietal', 'Alpha3_Occipital', 'Beta1_Parietal', 'Beta1_Occipital', 'Theta_Frontal', 'Alpha3_Temporal'],
                            relevant_features='all',
                            norm_method='min-max', augmentation=True
                            )
    """
    classification_with_var(run, pc, datasets_metadata_concatenated, _out_path=variation_out_path,
                            var_name="SITE",
                            model=RandomForestClassifier(n_estimators=200, max_depth=15, random_state=0),
                            relevant_features='all',
                            norm_method='min-max',  augmentation=True)
    """


