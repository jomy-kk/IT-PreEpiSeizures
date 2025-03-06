import pickle
from typing import Sequence

from imblearn.over_sampling import SMOTE
from neptune.types import File
from neptune.utils import stringify_unsupported
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import LeaveOneOut
from matplotlib import pyplot as plt
import seaborn as sns

from read import *

regions = ["Frontal", "Central", "Parietal", "Temporal", "Occipital", "Limbic"]
bands = ["Delta", "Theta", "Alpha1", "Alpha2", "Alpha3", "Beta1", "Beta2", "Gamma"]


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
