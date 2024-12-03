import itertools
from os.path import join

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
from matplotlib.colors import LinearSegmentedColormap, Normalize
from scipy.stats import norm, invgamma
from scipy.stats import shapiro
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import RFE
from sklearn.metrics import classification_report
from sklearn.model_selection import LeaveOneOut

image_format = "png"
dpi = 300

# Plot style
sns.set_style("whitegrid")
sns.set_palette("bright")

regions = ["Parietal", "Temporal", "Occipital", "Limbic"]
bands = ["Delta", "Theta", "Alpha1", "Alpha2", "Alpha3", "Beta1", "Beta2", "Gamma"]

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


def plot_babiloni_quality_control(in_path, out_path):
    # Load babilony_quality.pkl
    babilony_quality = pd.read_pickle(join(in_path, f"babilony_quality.pkl"))
    # What are the datasets names?
    datasets_names = set([k[1] for k in babilony_quality.keys()])
    # What are the diagnoses names?
    diagnoses_names = set([k[2] for k in babilony_quality.keys()])

    fig = plt.figure(figsize=(4 * 4, 6))

    # for each region -> a subplot
    for i, region in enumerate(regions):

        diagnosis_independency_intradataset_res = {}

        # Keep only all features of the region
        to_keep = [f"{band}_{region}" for band in bands]

        # Colors
        diagnoses_colors = {"HC": [100, 100, 100], "AD": [0, 74, 153]}
        diagnoses_colors = {l: [c / 255 for c in color] for l, color in diagnoses_colors.items()}

        # Line styles
        datasets_lines = {"Newcastle": "-", "Izmir": "--", "Istambul": "-."}

        # make subplot
        plt.subplot(2, 4, i + 1)

        # for each dataset -> a line
        for dataset_name in datasets_names:

            # for each diagnosis -> a color
            for i, D in enumerate(diagnoses_names):
                _mean_std = babilony_quality[(region, dataset_name, D)]
                mean, std = _mean_std['mean'], _mean_std['std']

                # Mean
                plt.plot(mean.index.to_numpy(), mean, label=D, linestyle=datasets_lines[dataset_name],
                         color=diagnoses_colors[D],
                         linewidth=1)

                # Std
                #plt.fill_between(y_mean.index, y_mean - y_std, y_mean + y_std, alpha=0.1, color=diagnoses_colors[D])

        plt.title(f"{region}")
        plt.ylabel('eLORETA Current Density')
        xlabels = [get_band_abbrev(band) for band in to_keep]
        plt.xticks(range(len(to_keep)), xlabels)
        plt.title(region)

    plt.subplots_adjust(hspace=0.3, wspace=0.3)
    plt.tight_layout()
    plt.savefig(join(out_path, f"babiloni_quality_control.{image_format}"), dpi=dpi, transparent=True, bbox_inches='tight')


def plot_mean_diffs(_datasets, outpath, log=False):
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
    plt.tight_layout()
    plt.savefig(join(outpath, f"mean_diffs.{image_format}"), dpi=dpi, transparent=True, bbox_inches='tight')


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


def plot_qq(_datasets, out_path):
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
        plt.savefig(join(out_path, f"qq_{dataset_name}.{image_format}"), dpi=dpi, transparent=True, bbox_inches='tight')


def plot_correlation_with_var(in_path, out_path, var_of_interest, variant):
    # x-axis: site/batch
    # y-axis: percentage of features significantly correlated with the variable of interest

    # Read correlations csv -> pandas dataframe
    correlations = pd.read_csv(join(in_path, f"correlation_{var_of_interest}.csv"), index_col=0)

    # Percentage of features significantly correlated with the variable of interest
    N_significant = sum(correlations['p'] < 0.05)
    significant = N_significant / len(correlations) * 100

    # plot
    plt.figure(figsize=(3, 4))
    plt.bar(variant, significant)
    plt.ylim(0, 100)
    plt.ylabel(f"% of features significantly correlated with {var_of_interest.capitalize()}")
    plt.tight_layout()
    plt.savefig(join(out_path, f"correlation_with_{var_of_interest}.{image_format}"), dpi=dpi, transparent=True, bbox_inches='tight')


def plot_classification_with_var(in_path, out_path, var_of_interest, variant):
    # x-axis: site/batch
    # y-axis: percentage of features significantly correlated with the variable of interest

    # Read classifications csv -> pandas dataframe
    classification_res = pd.read_csv(join(in_path, f"classification_{var_of_interest}.csv"), index_col=0)
    pred = classification_res['pred'].to_numpy()
    true = classification_res['true'].to_numpy()

    # Compute sensitivity and specificity from sklearn
    report = classification_report(true, pred, output_dict=True)
    print(var_of_interest)
    print(report)

    # plot
    plt.figure(figsize=(3, 4))
    plt.bar(variant, report['weighted avg']['f1-score'])
    plt.ylim(0, 1.0)
    plt.ylabel(f"F1-score (w.avg) to classify {var_of_interest.capitalize()}")
    plt.tight_layout()
    plt.savefig(join(out_path, f"classification_with_{var_of_interest}.{image_format}"), dpi=dpi, transparent=True, bbox_inches='tight')


def plot_2components(_metadata, in_path, out_path, method):
    # Read csv principal components file
    pc = pd.read_csv(join(in_path, f"{method}_transformed.csv"), index_col=0)

    # Plot stlyles
    dataset_colors = {"Newcastle": [62, 156, 73], "Izmir": [212, 120, 14], "Istambul": [182, 35, 50]}
    dataset_colors = {l: [c / 255 for c in color] for l, color in dataset_colors.items()}
    diagnoses_circles = {"HC": "x", "AD": "o"}

    # One plot, all datasets
    plt.figure(figsize=(6, 6))
    for dataset, color in dataset_colors.items():
        for diagnosis, marker in diagnoses_circles.items():
            idx = _metadata[(_metadata["DIAGNOSIS"] == diagnosis) & (_metadata["SITE"] == dataset)].index
            existing_idx = pc.index.intersection(idx)
            plt.scatter(pc['0'].loc[existing_idx], pc['1'].loc[existing_idx], color=color,
                        label=f"{dataset} - {diagnosis}", marker=marker)

    #plt.legend()
    plt.xlabel("Component 1", fontsize=14)
    plt.ylabel("Component 2", fontsize=14)
    plt.tight_layout()
    plt.savefig(join(out_path, f"{method}.{image_format}"), dpi=dpi, transparent=True, bbox_inches='tight')


def plot_distance_matrix(in_path, out_path):
    # Distance matrices show the Euclidean distance across all features between batch/site-average values
    # x-axis: batches
    # y-axis: batches

    # Read distance matrix txt -> numpy array
    with open(join(in_path, "distance_matrix.txt")) as f:
        matrix = np.loadtxt(f, dtype=str)
    dataset_names = matrix[0]
    matrix = matrix[1:].astype(float)

    # make my own colormap pastel where 0 is white and 1 is colour
    cmap = LinearSegmentedColormap.from_list("custom_cmap", ["white", "red"], N=256)
    plt.imshow(matrix, cmap=cmap, norm=Normalize(vmin=0, vmax=3))
    for i in range(len(matrix)):
        for j in range(len(matrix)):
            plt.text(j, i, f"{matrix[i][j]:.2f}", ha='center', va='center', color='black')
    plt.grid(False)
    plt.xticks(range(len(matrix)), dataset_names, rotation=0)
    plt.yticks(range(len(matrix)), dataset_names)
    avg_distance = np.mean([v for v in np.array(matrix).flatten() if v > 0])
    plt.title(f"Distance between datasets\nAverage distance: {avg_distance:.2f}")
    plt.tight_layout()
    plt.savefig(join(out_path, f"distance_matrix.{image_format}"), dpi=dpi, transparent=True, bbox_inches='tight')


def plot_batch_effects_dist(in_path, out_path):
    print("Creating batch effects distribution plots.")
    # Load distributions_by_dataset.pkl
    distributions = pd.read_pickle(join(in_path, "distributions_by_dataset.pkl"))

    # visualize fit of the prior distribution, along with the observed distribution of site effects
    colors = ['blue', 'red', 'green']

    # Gamma prior and observed
    plt.figure()
    for i, dataset_name in enumerate(distributions.keys()):
        dataset_dist = distributions[dataset_name]
        gamma_hat = dataset_dist['gamma_hat']
        normal_dist = dataset_dist['normal_dist']
        sns.kdeplot(normal_dist, color=colors[i], label=f'{dataset_name} Prior', linestyle='--')
        sns.kdeplot(gamma_hat, color=colors[i], label=f'{dataset_name} Observed', linestyle='-')
    plt.legend()
    plt.title("Additive Batch Effects (Gamma)")
    plt.tight_layout()
    plt.savefig(join(out_path, f"add_batch_effects_dist.{image_format}"), dpi=dpi, transparent=True, bbox_inches='tight')

    # Delta squared prior and observed
    plt.figure()
    for i, dataset_name in enumerate(distributions.keys()):
        dataset_dist = distributions[dataset_name]
        inverse_gamma_dist = dataset_dist['inverse_gamma_dist']
        delta_hat = dataset_dist['delta_hat']
        sns.kdeplot(inverse_gamma_dist, color=colors[i], label=f'{dataset_name} Prior', linestyle='--')
        sns.kdeplot(delta_hat, color=colors[i], label=f'{dataset_name} Observed', linestyle='-')
    plt.legend()
    plt.title("Multiplicative Batch Effects (Delta^2)")
    plt.tight_layout()
    plt.savefig(join(out_path, f"mul_batch_effects_dist.{image_format}"), dpi=dpi, transparent=True, bbox_inches='tight')
