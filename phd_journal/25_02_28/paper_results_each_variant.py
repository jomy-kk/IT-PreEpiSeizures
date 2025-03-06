from os import makedirs
from os.path import exists

import pandas as pd
from neptune.types import File
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import ElasticNet
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier

from processing import *
from read import *

#############

def make_results(run, out_path, dataset_names, harmonization_method, cov_age, cov_gender, cov_education, cov_diagnosis, MMSE_criteria, MoCA_criteria, n_pc, model):
    model.random_state = 0  # fix seed

    variation_out_path = join(out_path, str(model))
    makedirs(variation_out_path, exist_ok=True)

    # Read metadata
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

    run["datasets/criteria"] = f"AD criteria: AD & (MMSE<={MMSE_criteria[0]} | MoCA<={MoCA_criteria[0]})\n" \
                                f"HC criteria: HC & (MMSE>={MMSE_criteria[1]} | MoCA>={MoCA_criteria[1]})"

    # Get already existing harmonized data
    datasets = {}
    if exists(harmonization_method):
        print(f"Utilizing harmonized serialised data")
        for dataset_name in dataset_names:
            harmonized_data_path = join(harmonization_method, f"{dataset_name}.csv")
            print("Loading", harmonized_data_path)
            datasets[dataset_name] = pd.read_csv(harmonized_data_path, index_col=0)
            run[f"datasets/after/{dataset_name}_track"].track_files(harmonized_data_path)
            run[f"datasets/after/{dataset_name}"].upload(harmonized_data_path)
    else:
        raise Exception("No harmonized data found")

    run["harmonization/params"] = {
        "method": harmonization_method if harmonization_method != "neuroharmonize" else "neuroharmonize2",  # To fix for the fact that in Neptune, 'neuroharmonize' is in fact 'neurocombat' and 'neuroharmonize2' is the true 'neuroharmonize'
        "log_transform": True,
        "standardize": 'none',
        "cov_age": cov_age,
        "cov_gender": cov_gender,
        "cov_education": cov_education,
        "cov_diagnosis": cov_diagnosis
    }

    # --- Process data for plots ---

    # Concatenate datasets
    datasets_concatenated = pd.concat(datasets.values(), axis=0)
    datasets_metadata_concatenated = pd.concat(datasets_metadata.values(), axis=0)
    datasets_metadata_concatenated = datasets_metadata_concatenated.loc[datasets_concatenated.index]
    assert len(datasets_concatenated) == len(datasets_metadata_concatenated) == 276

    # PCA on features
    pc = simple_diagnosis_discriminant(run, datasets_concatenated, datasets_metadata_concatenated, variation_out_path,
                                  method='pca', norm_method='none', n_components=n_pc,
                                  relevant_features='all'
                                  )

    # Classification with PCs
    classification_with_var(run, pc, datasets_metadata_concatenated, _out_path=variation_out_path,
                            var_name="DIAGNOSIS",
                            model=model, relevant_features='all',
                            norm_method='min-max', augmentation=True
                            )
