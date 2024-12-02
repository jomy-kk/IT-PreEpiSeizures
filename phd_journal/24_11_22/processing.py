import itertools
import pickle

from scipy.stats import pearsonr
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE
from sklearn.manifold import TSNE
from sklearn.model_selection import LeaveOneOut

from combat_variations import *
from read import *

regions = ["Frontal", "Central", "Parietal", "Temporal", "Occipital", "Limbic"]
bands = ["Delta", "Theta", "Alpha1", "Alpha2", "Alpha3", "Beta1", "Beta2", "Gamma"]



# 0. Apply Combat
def apply_combat(datasets, datasets_metadata, log_transform, harmonization_method, cov_age, cov_gender, cov_education, cov_diagnosis):
    if log_transform:
        # Approximate normality by log transformation
        datasets = {dataset_name: intra_dataset_norm(dataset, method='log') for dataset_name, dataset in datasets.items()}

    # Indexes of each dataset
    indexes = {dataset_name: datasets[dataset_name].index for dataset_name in datasets.keys()}

    # Join all datasets and metadata
    X = pd.concat(datasets.values())
    all_metadata = pd.concat(datasets_metadata.values())
    all_metadata = all_metadata.loc[X.index]  # keep only the metadata of the subjects in X
    assert X.shape[0] == all_metadata.shape[0]

    # Apply Combat
    match harmonization_method:
        case "none":
            dist_parameters = None
        case "original":
            X, dist_parameters = original_combat(X, all_metadata, cov_age=cov_age,
                                                        cov_gender=cov_gender, cov_education=cov_education, cov_diagnosis=cov_diagnosis)
        case "neurocombat":
            X, dist_parameters = neuro_combat(X, all_metadata, cov_age=cov_age,
                                                     cov_gender=cov_gender, cov_education=cov_education, cov_diagnosis=cov_diagnosis)
        case "neuroharmonize":
            X, dist_parameters = neuro_harmonize(X, all_metadata, cov_age=cov_age,
                                                        cov_gender=cov_gender, cov_education=cov_education, cov_diagnosis=cov_diagnosis)
        case _:
            raise ValueError(f"Unknown harmonization method: {harmonization_method}")

    # Split the datasets and undo log transformation
    return {dataset_name: np.power(10, X.loc[indexes[dataset_name].intersection(X.index)]) for dataset_name in datasets.keys()}, dist_parameters


# 1. Batch effects distribution
def batch_effects_distribution(dist_parameters, _out_path):
    if dist_parameters is not None:
        distributions_by_dataset = {}
        for i, dataset_name in enumerate(dist_parameters['sites']):
            normal_dist = norm.rvs(size=10000, loc=dist_parameters['gamma_bar'][i], scale=np.sqrt(dist_parameters['t2'][i]), random_state=42)
            gamma_hat = dist_parameters['gamma_hat'][i]
            inverse_gamma_dist = invgamma.rvs(a=dist_parameters['a_prior'][i], scale=dist_parameters['b_prior'][i], size=10000)
            delta_hat = dist_parameters['delta_hat'][i]
            distributions_by_dataset[dataset_name] = {'normal_dist': normal_dist, 'gamma_hat': gamma_hat, 'inverse_gamma_dist': inverse_gamma_dist, 'delta_hat': delta_hat}
        with open(join(_out_path, "distributions_by_dataset.pkl"), "wb") as f:
            pickle.dump(distributions_by_dataset, f)


# 2. Q-Q plots
# Noting to prepare

# 3. Mean-diffs
def mean_diffs(datasets, _out_path):
    all_dataset_names = list(datasets.keys())
    combinations = list(itertools.combinations(all_dataset_names, 2))
    res = {}
    for i, (dataset_name_1, dataset_name_2) in enumerate(combinations):
        if dataset_name_1 == dataset_name_2:
            break
        # Extract datasets for Izmir and Newcastle
        d1_data = datasets[dataset_name_1]
        d2_data = datasets[dataset_name_2]
        # Calculate the difference and average
        diff_values = d1_data.mean() - d2_data.mean()
        avg_values = (d1_data.mean() + d2_data.mean()) / 2
        res[dataset_name_1, dataset_name_2] = {'diff': diff_values, 'avg': avg_values}
    # Serialize res
    with open(join(_out_path, "mean_diffs.pkl"), "wb") as f:
        pickle.dump(res, f)


# 4. Babiloni quality control
def babiloni_quality_control(datasets, datasets_metadata, _out_path):
    res = {}
    for i, region in enumerate(regions):
        diagnosis_independency_intradataset_res = {}
        # Keep only all features of the region
        to_keep = [f"{band}_{region}" for band in bands]
        for dataset_name, metadata in datasets_metadata.items():
            dataset = datasets[dataset_name]
            dataset = dataset[to_keep]
            # Get unique diagnoses
            diagnoses = list(metadata['DIAGNOSIS'].unique())
            for i, D in enumerate(diagnoses):
                # Get all subjects with that diagnosis
                y_metadata = metadata[metadata['DIAGNOSIS'] == D]
                subjects = y_metadata.index.tolist()
                existing_subjects = dataset.index.intersection(subjects)
                y = dataset.loc[existing_subjects]
                # Average each feature across subjects
                y_mean = y.mean(axis=0)
                y_std = y.std(axis=0)
                res[(region, dataset_name, D)] = {'mean': y_mean, 'std': y_std}
    # Serialize res
    with open(join(_out_path, "babilony_quality.pkl"), "wb") as f:
        pickle.dump(res, f)


# 5. tSNE and LDA
def tsne_lda(datasets_concatenated, datasets_metadata_concatenated, _out_path):
    # Linear Discriminant Analysis
    lda = LDA(n_components=2)
    _metadata = datasets_metadata_concatenated.loc[datasets_metadata_concatenated.index.intersection(datasets_concatenated.index)]
    lda.fit(datasets_concatenated, _metadata["SITE"])
    lda_transformed = lda.transform(datasets_concatenated)
    lda_transformed = pd.DataFrame(lda_transformed, index=datasets_concatenated.index)
    # Save model and transformed
    with open(join(_out_path, "lda_model.pkl"), "wb") as f:
        pickle.dump(lda, f)
    lda_transformed.to_csv(join(_out_path, "lda_transformed.csv"))

    # tSNE
    tsne = TSNE(n_components=2)
    tsne_transformed = tsne.fit_transform(datasets_concatenated)
    tsne_transformed = pd.DataFrame(tsne_transformed, index=datasets_concatenated.index)
    # Save model and transformed
    with open(join(_out_path, "tsne_model.pkl"), "wb") as f:
        pickle.dump(tsne, f)
    tsne_transformed.to_csv(join(_out_path, "tsne_transformed.csv"))


# 6. Distance Matrix
def distance_matrix(datasets, _out_path):
    matrix = [[0] * len(datasets) for _ in range(len(datasets))]
    for i, d1_name in enumerate(datasets.keys()):
        for j, d2_name in enumerate(datasets.keys()):
            if i > j:
                d1_data = datasets[d1_name]
                d2_data = datasets[d2_name]
                distance = ((d1_data.mean() - d2_data.mean()) ** 2).sum() ** 0.5  # Euclidean distance
                matrix[i][j] = distance

    # open a file in write mode
    with open(join(_out_path, "distance_matrix.txt"), "w") as f:
        # first line is dataset names
        f.write(" ".join(datasets.keys()) + "\n")
        # write the matrix
        for row in matrix:
            f.write(" ".join(map(str, row)) + "\n")


# 7. Correlation with var
def correlation_with_var(datasets_concatenated, datasets_metadata_concatenated, _out_path):
    # By Diagnosis and Site
    for var_name in ("DIAGNOSIS", "SITE"):
        metadata_var = datasets_metadata_concatenated[var_name]
        if var_name == "DIAGNOSIS":
            metadata_var.replace({"HC": 0, "AD": 1}, inplace=True)
            metadata_var = metadata_var.astype(int)
        if var_name == "SITE":
            metadata_var.replace({"Newcastle": 0, "Izmir": 1, "Istambul": 2}, inplace=True)
            metadata_var = metadata_var.astype(int)
        metadata_var = metadata_var[datasets_concatenated.index]
        # Calculate correlation for each dataset
        r, p = [], []
        for column in datasets_concatenated.columns:
            r_, p_ = pearsonr(datasets_concatenated[column], metadata_var)
            r.append(r_)
            p.append(p_)
        # Make df with r and p in rows and columns as features
        df = pd.DataFrame({"r": r, "p": p}, index=datasets_concatenated.columns)
        # Save df
        df.to_csv(join(_out_path, f"correlation_{var_name}.csv"))


# 8. Classification with var
def classification_with_var(datasets_concatenated, datasets_metadata_concatenated, _out_path):
    # By Diagnosis and Site
    for var_name in ("DIAGNOSIS", "SITE"):
        metadata_var = datasets_metadata_concatenated[var_name]
        if var_name == "DIAGNOSIS":
            metadata_var.replace({"HC": 0, "AD": 1}, inplace=True)
            metadata_var = metadata_var.astype(int)
        if var_name == "SITE":
            metadata_var.replace({"Newcastle": 0, "Izmir": 1, "Istambul": 2}, inplace=True)
            metadata_var = metadata_var.astype(int)
        metadata_var = metadata_var[datasets_concatenated.index]

        # RFE
        model = RandomForestClassifier(n_estimators=150, max_depth=10, random_state=0)
        rfe = RFE(estimator=model, n_features_to_select=10, step=1)
        rfe.fit(datasets_concatenated, metadata_var)
        relevant_features = datasets_concatenated.columns[rfe.support_]
        #print("Relevant features by RFE:")
        #print(relevant_features)
        # Save relevant features in txt
        with open(join(_out_path, f"relevant_features_{var_name}.txt"), "w") as f:
            f.write("\n".join(relevant_features))

        # Select only relevant features
        selected_features = datasets_concatenated[relevant_features]

        loo = LeaveOneOut()
        index, pred, true = [], [], []
        for i, (train_index, test_index) in enumerate(loo.split(datasets_concatenated)):
            model = RandomForestClassifier(n_estimators=200, max_depth=15, random_state=0)
            X_train, X_test = selected_features.iloc[train_index], selected_features.iloc[test_index]
            y_train, y_test = metadata_var.iloc[train_index], metadata_var.iloc[test_index]

            # Make sample weights (n_samples, ) according to the class size
            classes = y_train.unique()
            class_weight = {c: len(y_train) / (len(classes) * len(y_train[y_train == c])) for c in classes}
            sample_weight = y_train.map(class_weight)

            # Train with weights
            model.fit(X_train, y_train, sample_weight=sample_weight)

            # Validate
            y_pred = model.predict(X_test)[0]
            pred.append(y_pred)
            true.append(y_test.values[0])
            index.append(y_test.index[0])

        # make df with pred and true and save
        df = pd.DataFrame({"pred": pred, "true": true}, index=index)
        df.to_csv(join(_out_path, f"classification_{var_name}.csv"))
