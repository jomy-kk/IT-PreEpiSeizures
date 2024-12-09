import itertools
import pickle
from typing import Sequence

import numpy as np
from imblearn.over_sampling import SMOTE
from neptune.types import File
from neptune.utils import stringify_unsupported
from scipy.stats import pearsonr
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE
from sklearn.manifold import TSNE
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import LeaveOneOut
import neptune.integrations.sklearn as npt_utils

from combat_variations import *
from read import *


regions = ["Frontal", "Central", "Parietal", "Temporal", "Occipital", "Limbic"]
bands = ["Delta", "Theta", "Alpha1", "Alpha2", "Alpha3", "Beta1", "Beta2", "Gamma"]


# 0. Apply Combat
def apply_combat(run, datasets, datasets_metadata, _out_path, log_transform, standardize, harmonization_method, cov_age, cov_gender, cov_education, cov_diagnosis):
    match standardize:
        case "min-max":
            print("Normalizing each dataset before combat...")
            # Standardize each dataset
            means_stds = {}
            for dataset_name, dataset in datasets.items():
                min, max = dataset.min(), dataset.max()
                dataset = (dataset - min) / (max - min)
                datasets[dataset_name] = dataset
                means_stds[dataset_name] = {'min': min, 'max': max}
        case "z-score":
            print("Standardizing each dataset before combat...")
            # Standardize each dataset
            means_stds = {}
            for dataset_name, dataset in datasets.items():
                mean, std = dataset.mean(), dataset.std()
                dataset = (dataset - mean) / std
                datasets[dataset_name] = dataset
                means_stds[dataset_name] = {'mean': mean, 'std': std}


    if log_transform:
        # Approximate normality by log transformation
        # Note: log(0) is undefined, so we add 1 to all values
        datasets = {dataset_name: intra_dataset_norm(dataset+1, method='log') for dataset_name, dataset in datasets.items()}

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
            print(0)
            dist_parameters = None
        case "original":
            print(1)
            X = original_combat(X, all_metadata, cov_age=cov_age, cov_gender=cov_gender, cov_education=cov_education, cov_diagnosis=cov_diagnosis)
            dist_parameters = None
        case "neurocombat":
            print(2)
            X, dist_parameters = neuro_combat(X, all_metadata, cov_age=cov_age,
                                                     cov_gender=cov_gender, cov_education=cov_education, cov_diagnosis=cov_diagnosis)
        case "neuroharmonize":
            print(3)
            X, dist_parameters = neuro_harmonize(X, all_metadata, cov_age=cov_age,
                                                        cov_gender=cov_gender, cov_education=cov_education, cov_diagnosis=cov_diagnosis)
        case _:
            raise ValueError(f"Unknown harmonization method: {harmonization_method}")

    # Split the datasets and undo log transformation
    res = {dataset_name: X.loc[indexes[dataset_name].intersection(X.index)] for dataset_name in datasets.keys()}
    if log_transform:
        # Undo log transformation and +1
        res = {dataset_name: np.exp(res[dataset_name]) - 1 for dataset_name in res.keys()}
    match standardize:
        case "min-max":
            # Undo standardization
            res = {dataset_name: res[dataset_name] * (means_stds[dataset_name]['max'] - means_stds[dataset_name]['min']) + means_stds[dataset_name]['min'] for dataset_name in res.keys()}
        case "z-score":
            # Undo standardization
            res = {dataset_name: res[dataset_name] * means_stds[dataset_name]['std'] + means_stds[dataset_name]['mean'] for dataset_name in res.keys()}

    run["harmonization/params"] = {
        "method": harmonization_method,
        "log_transform": log_transform,
        "standardize": standardize,
        "cov_age": cov_age,
        "cov_gender": cov_gender,
        "cov_education": cov_education,
        "cov_diagnosis": cov_diagnosis
    }

    return res, dist_parameters


# 1. Batch effects distribution
def batch_effects_distribution(run, dist_parameters, _out_path):
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

        run["harmonization/batch_effects_dist"] = stringify_unsupported(distributions_by_dataset)


# 2. Q-Q plots
# Noting to prepare

# 3. Mean-diffs
def mean_diffs(run, datasets, _out_path):
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
def babiloni_quality_control(run, datasets, datasets_metadata, _out_path):
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
                res[(region, dataset_name, D)] = y
    # Serialize res
    with open(join(_out_path, "babilony_quality.pkl"), "wb") as f:
        pickle.dump(res, f)


# 5. tSNE and LDA
def tsne_lda(run, datasets_concatenated, datasets_metadata_concatenated, _out_path):
    # Linear Discriminant Analysis
    try:
        lda = LDA(n_components=2, priors=None, shrinkage=None, solver='svd', store_covariance=False, tol=0.0001)
        _metadata = datasets_metadata_concatenated.loc[datasets_metadata_concatenated.index.intersection(datasets_concatenated.index)]
        lda.fit(datasets_concatenated, _metadata["SITE"])
        lda_transformed = lda.transform(datasets_concatenated)
        lda_transformed = pd.DataFrame(lda_transformed, index=datasets_concatenated.index)
        # Save model and transformed
        with open(join(_out_path, "lda_model.pkl"), "wb") as f:
            pickle.dump(lda, f)
        lda_transformed.to_csv(join(_out_path, "lda_transformed.csv"))
    except ValueError:
        pass

    # tSNE
    tsne = TSNE(n_components=2, learning_rate='auto', perplexity=30, random_state=0)
    tsne_transformed = tsne.fit_transform(datasets_concatenated)
    tsne_transformed = pd.DataFrame(tsne_transformed, index=datasets_concatenated.index)
    # Save model and transformed
    with open(join(_out_path, "tsne_model.pkl"), "wb") as f:
        pickle.dump(tsne, f)
    tsne_transformed.to_csv(join(_out_path, "tsne_transformed.csv"))


# 6. Distance Matrix
def distance_matrix(run, datasets, _out_path):
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
def correlation_with_var(run, datasets_concatenated, datasets_metadata_concatenated, _out_path):
    # By Diagnosis and Site
    for var_name in ("DIAGNOSIS", "SITE"):
        metadata_var = datasets_metadata_concatenated[var_name]
        metadata_var = pd.Series(pd.factorize(metadata_var)[0], index=metadata_var.index)
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
def classification_with_var(run, datasets_concatenated, datasets_metadata_concatenated, _out_path,
                            var_name, model, relevant_features=None, norm_method=None, augmentation=False):
    # Normalise/Standardize
    match norm_method:
        case "z-score":
            print("Applying z-score...")
            datasets_concatenated = (datasets_concatenated - datasets_concatenated.mean()) / datasets_concatenated.std()
        case "min-max":
            #print("Applying min-max norm...")
            datasets_concatenated = (datasets_concatenated - datasets_concatenated.min()) / (datasets_concatenated.max() - datasets_concatenated.min())
        case None:
            print("No normalization applied.")
            pass

    run[f"classification/{var_name}/norm"] = norm_method

    # Shuffle the data
    datasets_concatenated = datasets_concatenated.sample(frac=1, random_state=0)

    #print(f"Classification with {var_name}")
    datasets_metadata_concatenated = datasets_metadata_concatenated.loc[datasets_concatenated.index]
    metadata_var = datasets_metadata_concatenated[var_name]
    metadata_var = pd.Series(pd.factorize(metadata_var)[0], index=metadata_var.index)
    #print("Targets:", metadata_var.unique())
    metadata_var = metadata_var[datasets_concatenated.index]

    #print(model.get_params())

    if isinstance(relevant_features, int):
        run[f"classification/{var_name}/relevant_features"] = f"RFE ({relevant_features})"
        # RFE
        print("Running RFE...")
        model_rfe = RandomForestClassifier(n_estimators=200, max_depth=15, random_state=0)
        rfe = RFE(estimator=model_rfe, n_features_to_select=relevant_features, step=1)

        # Make sample weights (n_samples, ) according to the class size
        classes = metadata_var.unique()
        class_weight = {c: len(metadata_var) / (len(classes) * len(metadata_var[metadata_var == c])) for c in classes}
        sample_weight = metadata_var.map(class_weight)

        rfe.fit(datasets_concatenated, metadata_var, sample_weight=sample_weight)
        relevant_features = datasets_concatenated.columns[rfe.support_]
        # Save relevant features in txt
        with open(join(_out_path, f"relevant_features_{var_name}.txt"), "w") as f:
            f.write("\n".join(relevant_features))

        # Plot feature importances
        importances = rfe.estimator_.feature_importances_
        indices = np.argsort(importances)[::-1]
        plt.close()
        fig = plt.figure()
        plt.title("Feature importances")
        plt.bar(range(len(relevant_features)), importances[indices], align="center")
        plt.xticks(range(len(relevant_features)), relevant_features, rotation=90)
        plt.xlim([-1, datasets_concatenated.shape[1]])
        run[f"classification/{var_name}/feature_importance"] = File.as_image(fig)
        plt.close(fig)

    # Select only relevant features
    if isinstance(relevant_features, str) and relevant_features == 'all':
        selected_features = datasets_concatenated
    else:
        selected_features = datasets_concatenated[relevant_features]
    #print("Selected features:")
    #print(selected_features.columns)
    run[f"classification/{var_name}/features"] = stringify_unsupported(selected_features.columns.tolist())

    loo = LeaveOneOut()
    index, pred, true = [], [], []
    for i, (train_index, test_index) in enumerate(loo.split(datasets_concatenated)):
        X_train, X_test = selected_features.iloc[train_index], selected_features.iloc[test_index]
        y_train, y_test = metadata_var.iloc[train_index], metadata_var.iloc[test_index]

        # Shuffle X_train and keep the same order for y_train
        X_train = X_train.sample(frac=1, random_state=0)
        y_train = y_train.loc[X_train.index]
        assert (X_train.index == y_train.index).all()

        # Make sample weights (n_samples, ) according to the class size
        if augmentation:
            # SMOTE
            smote = SMOTE(k_neighbors=5, random_state=0)
            X_train, y_train = smote.fit_resample(X_train, y_train)
        else:
            classes = y_train.unique()
            class_weight = {c: len(y_train) / (len(classes) * len(y_train[y_train == c])) for c in classes}
            sample_weight = y_train.map(class_weight)

        # Create fresh model with same architecture
        model = model.__class__(**model.get_params())

        # Train with weights
        if not augmentation:
            try:
                run[f"classification/{var_name}/class_weight"] = str(class_weight)
                model.fit(X_train, y_train, sample_weight=sample_weight)
                print(f"LOOCV {i}/{len(metadata_var)}. Used class weights.")
            except TypeError:
                run[f"classification/{var_name}/class_weight"] = "n.a."
                model.fit(X_train, y_train)
                print(f"LOOCV {i}/{len(metadata_var)}. Did not use class weights.")
        else:
            run[f"classification/{var_name}/class_weight"] = "SMOTE augmented"
            model.fit(X_train, y_train)
            #print(f"LOOCV {i}/{len(metadata_var)}. SMOTE augmented.")

        # Validate
        y_pred = model.predict(X_test)[0]
        # Is y_pred a probability?
        #print(y_pred)
        if isinstance(y_pred, Sequence) and len(y_pred) > 1:
            y_pred = np.argmax(y_pred)
            run[f"classification/{var_name}/output_prob"] = True
        else:
            run[f"classification/{var_name}/output_prob"] = False
        pred.append(y_pred)
        true.append(y_test.values[0])
        index.append(y_test.index[0])

        # Go updating score in the terminal
        f1 = f1_score(true, pred, average='weighted')
        #print(f"F1 score: {f1:.2f}")

    run[f"classification/{var_name}/model_description"] = str(model)
    run[f"classification/{var_name}/validation"] = str(loo)

    # make df with pred and true and save
    df = pd.DataFrame({"pred": pred, "true": true}, index=index)
    df.to_csv(join(_out_path, f"classification_{var_name}.csv"))
    run[f"classification/{var_name}/test_preds"].upload(join(_out_path, f"classification_{var_name}.csv"))


def simple_diagnosis_discriminant(run, datasets_concatenated, datasets_metadata_concatenated, _out_path, method='lda', n_components=2,
                                  norm_method='min-max', relevant_features=None):
    # Normalise/Standardize
    match norm_method:
        case "z-score":
            print("Applying z-score...")
            datasets_concatenated = (datasets_concatenated - datasets_concatenated.mean()) / datasets_concatenated.std()
        case "min-max":
            #print("Applying min-max norm...")
            datasets_concatenated = (datasets_concatenated - datasets_concatenated.min()) / (
                        datasets_concatenated.max() - datasets_concatenated.min())
        case None:
            print("No normalization applied.")
            pass
    run[f"simple_diagnosis_discriminant/norm"] = norm_method

    # Select only relevant features
    if isinstance(relevant_features, str) and relevant_features == 'all':
        datasets_concatenated = datasets_concatenated
    else:
        datasets_concatenated = datasets_concatenated[relevant_features]
    #print("Features for dimensionality reduction:")
    #print(datasets_concatenated.columns.tolist())
    run[f"simple_diagnosis_discriminant/features"] = stringify_unsupported(datasets_concatenated.columns.tolist())

    # Shuffle the data
    datasets_concatenated = datasets_concatenated.sample(frac=1, random_state=0)
    if method == 'lda':  # Linear Discriminant Analysis
        class_priors = datasets_metadata_concatenated["DIAGNOSIS"].value_counts(normalize=True)
        run[f"simple_diagnosis_discriminant/class_weight"] = str(class_priors)

        lda = LDA(n_components=n_components, priors=class_priors, shrinkage=None, solver='svd', store_covariance=False, tol=0.0001)
        run[f"simple_diagnosis_discriminant/model_description"] = str(lda)
        _metadata = datasets_metadata_concatenated.loc[datasets_metadata_concatenated.index.intersection(datasets_concatenated.index)]
        lda.fit(datasets_concatenated, _metadata["DIAGNOSIS"])
        lda_transformed = lda.transform(datasets_concatenated)
        lda_transformed = pd.DataFrame(lda_transformed, index=datasets_concatenated.index)
        # Save model and transformed
        with open(join(_out_path, f"simple_diagnosis_discriminant.pkl"), "wb") as f:
            pickle.dump(lda, f)
        lda_transformed.to_csv(join(_out_path, f"simple_diagnosis_discriminant_transformed.csv"))

        # Classify with LDA and get F1 score
        y_pred = lda.predict(datasets_concatenated)
        y_true = _metadata["DIAGNOSIS"]
        df = pd.DataFrame({"pred": y_pred, "true": y_true}, index=y_true.index)
        df.to_csv(join(_out_path, f"simple_diagnosis_discriminant.csv"))
        run["simple_diagnosis_discriminant/test_preds"].upload(join(_out_path, f"simple_diagnosis_discriminant.csv"))

        # Save report
        report = classification_report(y_true, y_pred, output_dict=True)
        print(f"Simple diagnosis discriminant report - {method}({n_components}):")
        print(report)
        run["simple_diagnosis_discriminant/my_metrics"] = report

        # Confusion matrix
        fig = plt.figure(figsize=(4, 4))
        sns.heatmap(confusion_matrix(y_true, y_pred), annot=True, fmt='d', cmap='Blues', cbar=False)
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.tight_layout()
        run[f"simple_diagnosis_discriminant/confusion_matrix"].upload(File.as_image(plt.gcf()))
        plt.savefig(join(_out_path, "simple_diagnosis_discriminant_confusion_matrix.pdf"), dpi=300, transparent=True,
                    bbox_inches='tight')
        plt.close(fig)

    elif method == 'pca':
        pca = PCA(n_components=n_components, random_state=0, svd_solver='auto')
        run[f"simple_diagnosis_discriminant/model_description"] = str(pca)
        pca.fit(datasets_concatenated)
        pca_transformed = pca.transform(datasets_concatenated)
        pca_transformed = pd.DataFrame(pca_transformed, index=datasets_concatenated.index)
        # Save model and transformed
        with open(join(_out_path, f"simple_diagnosis_discriminant.pkl"), "wb") as f:
            pickle.dump(pca, f)
        pca_transformed.to_csv(join(_out_path, f"simple_diagnosis_discriminant_transformed.csv"))
        return pca_transformed
    else:
        raise ValueError("Invalid method")
