import itertools
from os.path import join

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
from matplotlib.colors import LinearSegmentedColormap, Normalize
from scipy.stats import shapiro, pearsonr
from scipy.stats import ttest_ind
from sklearn.decomposition import PCA
# import RF classifier
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.feature_selection import RFE
# import roc_auc_score
from sklearn.metrics import roc_auc_score
# import leave one out cross validation
from sklearn.model_selection import LeaveOneOut

plots_path = "/Users/saraiva/Desktop/Doktorand/Combat Project"

# Plot style
sns.set_style("whitegrid")
sns.set_palette("bright")

regions = ["Frontal", "Central", "Parietal", "Temporal", "Occipital", "Limbic"]

# Set default font size
plt.rcParams.update({'font.size': 13})
# Set default Verdana font
plt.rcParams.update({'font.family': 'Verdana'})
# Set default colour for text
#plt.rcParams.update({'text.color': [0/255, 37/255, 77/255]})


def get_band_abbrev(band: str) -> str:
    band = band.split("_")[0]
    band = band.replace("Delta", "δ")
    band = band.replace("Theta", "θ")
    band = band.replace("Alpha", "α")
    band = band.replace("Beta", "β")
    band = band.replace("Gamma", "γ")
    return band


def plot_mean_std_indep(_datasets, _datasets_metadata, log_scale=False):
    if not log_scale:
        fig = plt.figure(figsize=(4 * 3, 3 * 2))
    else:
        plt.figure(figsize=(6, 4))  # Adjust the figure size to be smaller
    for i, region in enumerate(regions):

        diagnosis_independency_intradataset_res = {}

        # Keep only all features of the region
        to_keep = [f"{band}_{region}" for band in
                   ["Delta", "Theta", "Alpha1", "Alpha2", "Alpha3", "Beta1", "Beta2", "Gamma"]]

        # Colors
        diagnoses_colors = {"HC": [100, 100, 100], "AD": [0, 74, 153]}
        diagnoses_colors = {l: [c / 255 for c in color] for l, color in diagnoses_colors.items()}

        # Line styles
        datasets_lines = {"Newcastle": "-", "Izmir": "--", "Sapienza": "-."}

        # make subplot
        plt.subplot(2, 3, i + 1)

        for dataset_name, metadata in _datasets_metadata.items():
            dataset = _datasets[dataset_name]
            dataset = dataset[to_keep]

            # Get unique diagnoses
            diagnoses = list(metadata['DIAGNOSIS'].unique())

            Y_indep = []
            for i, D in enumerate(diagnoses):
                # Get all subjects with that diagnosis
                y_metadata = metadata[metadata['DIAGNOSIS'] == D]
                subjects = y_metadata.index.tolist()
                existing_subjects = dataset.index.intersection(subjects)
                y = dataset.loc[existing_subjects]
                Y_indep.append(y)

                # Average each feature across subjects
                y_mean = y.mean(axis=0)
                y_std = y.std(axis=0)

                # Line plot with error bars
                plt.plot(y_mean.index.to_numpy(), y_mean.to_numpy(), label=D, linestyle=datasets_lines[dataset_name],
                         color=diagnoses_colors[D],
                         linewidth=1)
                #plt.fill_between(y_mean.index, y_mean - y_std, y_mean + y_std, alpha=0.1, color=diagnoses_colors[D])

            # Compute t-test for each feature
            for band in to_keep:
                y1 = Y_indep[0][band].to_numpy()
                y2 = Y_indep[1][band].to_numpy()
                t, p = ttest_ind(y1, y2, equal_var=False)
                diagnosis_independency_intradataset_res[(dataset_name, band)] = p < 0.05
                #print(f"{dataset_name} {band} p-value: {p}")

        """
        # Add an asterisk if the diagnoses distributions are independent
        for band in to_keep:
            asterisks_label = ''
            for dataset_name in _datasets.keys():
                if diagnosis_independency_intradataset_res[(dataset_name, band)]:
                    asterisks_label += '*\n'

            # Draw asterisks directly above each band
            plt.text(to_keep.index(band), 0, asterisks_label, horizontalalignment='center', verticalalignment='bottom')
        """

        plt.title(f"{region}")
        #plt.ylabel('eLORETA Current Density')
        if log_scale:  # y in log scale?
            plt.yscale('log')
        xlabels = [get_band_abbrev(band) for band in to_keep]
        plt.xticks(range(len(to_keep)), xlabels)
        plt.title(region)

    plt.subplots_adjust(hspace=0.3, wspace=0.3)
    plt.savefig(join(plots_path, f"all_regions.pdf"), dpi=300, transparent=True, bbox_inches='tight')


def plot_mean_diffs(_datasets, log=False):
    # y-axis: difference between the average feature value at site 1 and the average feature value at site 2
    # x-axis: average feature value across all participants from both sites
    # only izmir and newcastle

    ymin, ymax = -6, 2
    if log:
        ymin, ymax = -1, 1

    all_dataset_names = list(_datasets.keys())
    combinations = list(itertools.combinations(all_dataset_names, 2))

    fig, axes = plt.subplots(1, 3, figsize=(4 * 3, 3))
    for i, (dataset_name_1, dataset_name_2) in enumerate(combinations):
        if dataset_name_1 == dataset_name_2:
            break

        # subplot
        ax = axes[i]

        # Extract datasets for Izmir and Newcastle
        d1_data = _datasets[dataset_name_1]
        d2_data = _datasets[dataset_name_2]

        # Calculate the difference and average
        diff_values = d1_data.mean() - d2_data.mean()
        avg_values = (d1_data.mean() + d2_data.mean()) / 2

        # Plotting
        ax.scatter(avg_values, diff_values, color='blue')
        ax.set_xlabel('Average feature value across sites')
        ax.set_ylabel('Feature difference across sites')
        ax.axhline(0, color='red', linestyle='--')
        ax.set_ylim(ymin, ymax)
        ax.set_title(f'{dataset_name_1} - {dataset_name_2}')

    plt.subplots_adjust(hspace=0.3, wspace=0.3)
    plt.show()


def check_normality(_datasets, _datasets_metadata):
    for dataset_name, dataset in _datasets.items():
        metadata = _datasets_metadata[dataset_name]
        print(f"Checking normality for dataset: {dataset_name}")
        N_normal = 0

        for column in dataset.columns:
            stat, p_value = shapiro(dataset[column])
            # print(f"{column}, {stat}, p-value={f'{p_value:.3e}'}", end=' ')
            if p_value > 0.05:
                # print(f"Normally distributed")# (fail to reject H0)")
                N_normal += 1
            else:
                pass
                # print(f"NOT normally distributed")# (reject H0)")

        print(f"Number of normally distributed features: {N_normal} out of {len(dataset.columns)}")


def create_qq_plots(_datasets):
    for dataset_name, dataset in _datasets.items():
        print(f"Creating Q-Q plots for dataset: {dataset_name}")
        num_features = len(dataset.columns)
        num_cols = 3
        num_rows = (num_features + num_cols - 1) // num_cols

        fig, axes = plt.subplots(num_rows, num_cols, figsize=(4*3, num_rows * 3))
        axes = axes.flatten()

        for i, column in enumerate(dataset.columns):
            sm.qqplot(dataset[column], line='s', ax=axes[i])
            axes[i].set_title(get_band_abbrev(column))

        for j in range(i + 1, len(axes)):
            fig.delaxes(axes[j])

        plt.tight_layout()
        plt.show()


def correlation_with_var(_datasets_before, _datasets_after, _metadata, vars_of_interest):
    # x-axis: site/batch
    # y-axis: percentage of features significantly correlated with the variable of interest

    # Concatenate _datasets_before
    _datasets_before = pd.concat(_datasets_before.values(), axis=0)
    _datasets_after = pd.concat(_datasets_after.values(), axis=0)
    _metadata = pd.concat(_metadata.values(), axis=0)

    relevant = []
    relevant_r = []

    for var_name in vars_of_interest:
        metadata_var = _metadata[var_name]

        if var_name == "DIAGNOSIS":
            metadata_var.replace({"HC": 0, "AD": 1}, inplace=True)
            metadata_var = metadata_var.astype(int)

        if var_name == "SITE":
            metadata_var.replace({"Newcastle": 0, "Izmir": 1, "Sapienza": 2}, inplace=True)
            metadata_var = metadata_var.astype(int)

        plt.figure(figsize=(3, 4))

        # Calculate correlation for each dataset
        r, p = [], []
        for version, dataset in {"Before": _datasets_before, "After": _datasets_after}.items():
            r_dataset, p_dataset = [], []
            for column in dataset.columns:
                r_, p_ = pearsonr(dataset[column], metadata_var[dataset.index])
                r_dataset.append(r_)
                p_dataset.append(p_)
                if var_name == "DIAGNOSIS" and p_ < 0.05:
                    relevant.append(column)
                    relevant_r.append(r_)

            r.append(r_dataset)
            p.append(p_dataset)

            # Percentage of features significantly correlated with the variable of interest
            N_significant = sum([p_ < 0.05 for p_ in p_dataset])
            significant = N_significant / len(p_dataset) * 100

            # plot
            plt.bar(version, significant)

        plt.ylim(0, 100)
        plt.ylabel(f"% of features significantly correlated with {var_name}")
        plt.show()

    # return features significantly correlated with 'DIAGNOSIS'
    # select 10 features with the highest correlation
    relevant = [x for _, x in sorted(zip(relevant_r, relevant), reverse=True)]
    return relevant


def classification_with_var(_datasets_before, _datasets_after, _metadata, vars_of_interest, relevant_features):
    # x-axis: site/batch
    # y-axis: percentage of features significantly correlated with the variable of interest

    # Concatenate _datasets_before
    _datasets_before = pd.concat(_datasets_before.values(), axis=0)
    _datasets_after = pd.concat(_datasets_after.values(), axis=0)
    _metadata = pd.concat(_metadata.values(), axis=0)

    for var_name in vars_of_interest:
        metadata_var = _metadata[var_name]

        if var_name == "DIAGNOSIS":
            metadata_var.replace({"HC": 0, "AD": 1}, inplace=True)
            metadata_var = metadata_var.astype(int)

        if var_name == "SITE":
            metadata_var.replace({"Newcastle": 0, "Izmir": 1, "Sapienza": 2}, inplace=True)
            metadata_var = metadata_var.astype(int)

        metadata_var = metadata_var[_datasets_before.index]

        plt.figure(figsize=(3, 4))

        # Linear Classifier with sklearn
        for version, dataset in {"Before": _datasets_before, "After": _datasets_after}.items():

            if relevant_features is None:  # RFE
                model = RandomForestClassifier(n_estimators=150, max_depth=10, random_state=0)
                # RFE
                rfe = RFE(estimator=model, n_features_to_select=10, step=1)
                rfe.fit(dataset, metadata_var)
                relevant_features = dataset.columns[rfe.support_]
                print("Relevant features by RFE:")
                print(relevant_features)
            else:
                relevant_features = relevant_features[:10]

            # Select only relevant features
            dataset = dataset[relevant_features]

            loo = LeaveOneOut()
            pred, true = [], []
            for i, (train_index, test_index) in enumerate(loo.split(dataset)):
                model = RandomForestClassifier(n_estimators=200, max_depth=15, random_state=0)
                X_train, X_test = dataset.iloc[train_index], dataset.iloc[test_index]
                y_train, y_test = metadata_var.iloc[train_index], metadata_var.iloc[test_index]

                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                pred.append(y_pred)
                true.append(y_test)

            # Compute sensitivity and specificity from sklearn
            from sklearn.metrics import classification_report
            report = classification_report(true, pred, output_dict=True)
            print(version)
            print(report)

            relevant_features = None

            # plot
            plt.bar(version, report['weighted avg']['f1-score'])

        plt.ylim(0, 1.0)
        plt.ylabel(f"F1-score (w.avg) to classify {var_name}")
        plt.show()


def regression_with_var(_datasets_before, _datasets_after, _metadata, vars_of_interest, relevant_features):
    # x-axis: site/batch
    # y-axis: percentage of features significantly correlated with the variable of interest

    # Concatenate _datasets_before
    _datasets_before = pd.concat(_datasets_before.values(), axis=0)
    _datasets_after = pd.concat(_datasets_after.values(), axis=0)
    _metadata = pd.concat(_metadata.values(), axis=0)

    for var_name in vars_of_interest:
        metadata_var = _metadata[var_name]

        if var_name == "MMSE":
            pass

        if var_name == "AGE":
            pass

        metadata_var = metadata_var[_datasets_before.index]

        # Drop nans
        metadata_var = metadata_var.dropna()
        _datasets_before = _datasets_before.loc[metadata_var.index]
        _datasets_after = _datasets_after.loc[metadata_var.index]

        plt.figure(figsize=(3, 4))

        # Linear Classifier with sklearn
        for version, dataset in {"Before": _datasets_before, "After": _datasets_after}.items():

            if relevant_features is None:  # RFE
                model = RandomForestRegressor(n_estimators=150, max_depth=10, random_state=0)
                # RFE
                rfe = RFE(estimator=model, n_features_to_select=10, step=1)
                rfe.fit(dataset, metadata_var)
                relevant_features = dataset.columns[rfe.support_]
                print("Relevant features by RFE:")
                print(relevant_features)
            else:
                relevant_features = relevant_features[:10]

            # Select only relevant features
            dataset = dataset[relevant_features]

            loo = LeaveOneOut()
            pred, true = [], []
            for i, (train_index, test_index) in enumerate(loo.split(dataset)):
                model = RandomForestRegressor(n_estimators=200, max_depth=15, random_state=0)
                X_train, X_test = dataset.iloc[train_index], dataset.iloc[test_index]
                y_train, y_test = metadata_var.iloc[train_index], metadata_var.iloc[test_index]

                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                pred.append(y_pred)
                true.append(y_test)

            # Compute rmse, pearson, r2
            from sklearn.metrics import mean_squared_error, r2_score
            rmse = mean_squared_error(true, pred, squared=False)
            #pearson = pearsonr(true, pred)
            r2 = r2_score(true, pred)
            print(version)
            print(f"RMSE: {rmse:.2f}")
            #print(f"Pearson: {pearson}")
            print(f"R2: {r2}")

            relevant_features = None

            # plot
            plt.bar(version, rmse)

        plt.ylim(0, 1.0)
        plt.ylabel(f"RMSE to regress {var_name}")
        plt.show()



def plot_2components(_datasets_before, _datasets_after, _metadata, method='pca'):
    # Concatenate _datasets_before
    _datasets_before = pd.concat(_datasets_before.values(), axis=0)
    _datasets_after = pd.concat(_datasets_after.values(), axis=0)
    _metadata = pd.concat(_metadata.values(), axis=0)

    # Find 2 principal components of each dataset
    if method == 'pca':
        before = PCA(n_components=2)
        after = PCA(n_components=2)
        before.fit(_datasets_before)
        after.fit(_datasets_after)
    elif method == 'lda':  # Linear Discriminant Analysis
        from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
        before = LDA(n_components=2)
        after = LDA(n_components=2)
        _metadata = _metadata.loc[_metadata.index.intersection(_datasets_before.index)]
        before.fit(_datasets_before, _metadata["SITE"])
        after.fit(_datasets_after, _metadata["SITE"])
    elif method == 'tsne':
        from sklearn.manifold import TSNE
        before = TSNE(n_components=2)
        after = TSNE(n_components=2)
        transformed_before = before.fit_transform(_datasets_before)
        transformed_after = after.fit_transform(_datasets_after)
    else:
        raise ValueError("Invalid method")

    # Transform data
    if method != 'tsne':
        transformed_before = before.transform(_datasets_before)
        transformed_after = after.transform(_datasets_after)
    transformed_before = pd.DataFrame(transformed_before, index=_datasets_before.index)
    transformed_after = pd.DataFrame(transformed_after, index=_datasets_after.index)

    # Plot stlyles
    dataset_colors = {"Newcastle": [62, 156, 73], "Izmir": [212, 120, 14], "Sapienza": [182, 35, 50]}
    dataset_colors = {l: [c / 255 for c in color] for l, color in dataset_colors.items()}
    diagnoses_circles = {"HC": "x", "AD": "o"}

    # One plot, all datasets
    for version, (model, original, pc) in {"Before": (before, _datasets_before, transformed_before), "After": (after, _datasets_after, transformed_after)}.items():
        plt.figure(figsize=(6, 6))
        for dataset, color in dataset_colors.items():
            for diagnosis, marker in diagnoses_circles.items():
                idx = _metadata[(_metadata["DIAGNOSIS"] == diagnosis) & (_metadata["SITE"] == dataset)].index
                existing_idx = pc.index.intersection(idx)
                plt.scatter(pc[0].loc[existing_idx], pc[1].loc[existing_idx], color=color,
                            label=f"{dataset} - {diagnosis}", marker=marker)

        # Calculate and plot discriminant lines
        if method == 'lda':
            x_min, x_max = original.iloc[:, 0].min() - 1, original.iloc[:, 0].max() + 1
            y_min, y_max = original.iloc[:, 1].min() - 1, original.iloc[:, 1].max() + 1
            xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))

            # Create a grid of points with the same number of features as the original data
            grid_points = np.c_[xx.ravel(), yy.ravel()]

            # Predict the class for each grid point
            Z = model.predict(grid_points)
            Z = Z.reshape(xx.shape)
            plt.contourf(xx, yy, Z, alpha=0.3, color='black')

        #plt.legend()
        plt.xlabel("Component 1", fontsize=14)
        plt.ylabel("Component 2", fontsize=14)
        plt.title(version)
        #plt.show()
        plt.savefig(join(plots_path, f"{method}_{version}.pdf"), dpi=300, transparent=True, bbox_inches='tight')


def plot_distance_matrix(_datasets_before, _datasets_after, _metadata):
    # Distance matrices show the Euclidean distance across all features between batch/site-average values
    # x-axis: batches
    # y-axis: batches

    for version, _datasets in {"Before": _datasets_before, "After": _datasets_after}.items():
        plt.figure(figsize=(4, 4))

        matrix = [[0] * len(_datasets) for _ in range(len(_datasets))]
        for i, d1_name in enumerate(_datasets.keys()):
            for j, d2_name in enumerate(_datasets.keys()):
                if i > j:
                    d1_data = _datasets[d1_name]
                    d2_data = _datasets[d2_name]
                    distance = ((d1_data.mean() - d2_data.mean()) ** 2).sum() ** 0.5  # Euclidean distance
                    matrix[i][j] = distance

        # plot matrix
        # make my own colormap pastel where 0 is white and 1 is colour
        cmap = LinearSegmentedColormap.from_list("custom_cmap", ["white", "red"], N=256)
        plt.imshow(matrix, cmap=cmap, norm=Normalize(vmin=0, vmax=3))
        for i in range(len(_datasets)):
            for j in range(len(_datasets)):
                plt.text(j, i, f"{matrix[i][j]:.2f}", ha='center', va='center', color='black')
        plt.grid(False)
        plt.xticks(range(len(_datasets)), _datasets.keys(), rotation=0)
        plt.yticks(range(len(_datasets)), _datasets.keys())
        avg_distance = np.mean([v for v in np.array(matrix).flatten() if v > 0])
        plt.title(f"Distance between datasets ({version})\nAverage distance: {avg_distance:.2f}")




