import itertools
from os.path import join, exists
from pickle import load

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
from matplotlib.colors import LinearSegmentedColormap, Normalize
from neptune.types import File
from scikitplot.metrics import plot_roc_curve
from scipy.stats import shapiro
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, roc_auc_score
import neptune.integrations.sklearn as npt_utils
import plotly.express as px


image_format = "png"
dpi = 300

# Plot style
sns.set_style("whitegrid")
sns.set_palette("bright")

regions = ["Parietal", "Temporal", "Occipital", "Limbic", "Frontal"]
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


def plot_babiloni_quality_control(run, in_path, out_path):
    # Load babilony_quality.pkl
    babilony_quality = pd.read_pickle(join(in_path, f"babilony_quality.pkl"))
    # What are the datasets names?
    datasets_names = set([k[1] for k in babilony_quality.keys()])
    # What are the diagnoses names?
    diagnoses_names = set([k[2] for k in babilony_quality.keys()])

    fig = plt.figure(figsize=(6 * len(regions), 6))

    # for each region -> a subplot
    for i, region in enumerate(regions):

        # Keep only all features of the region
        to_keep = [f"{band}_{region}" for band in bands]

        # Line styles
        diagnoses_lines = {"HC": '--', "AD": '-'}

        # Colors
        datasets_colors = ["blue", "green", "orange", "pink", "black", "yellow"]
        #datasets_colors = {l: [c / 255 for c in color] for l, color in datasets_colors.items()}

        # make subplot
        plt.subplot(1, len(regions), i + 1)

        diagnoses_shadow = {"HC": "grey", "AD": "red"}

        # for each diagnosis
        for i, D in enumerate(diagnoses_lines.keys()):

            # for each dataset
            for j, dataset_name in enumerate(datasets_names):

                y = babilony_quality[(region, dataset_name, D)]
                #mean, std = _mean_std['mean'], _mean_std['std']
                #q1, q2, q3 = stats['q1'], stats['q2'], stats['q3']
                q1, q2, q3 = y.quantile(0.25), y.quantile(0.5), y.quantile(0.75)

                # Mean
                #plt.plot(mean.index.to_numpy(), mean, linestyle=diagnoses_lines[D], color=datasets_colors[dataset_name], linewidth=1)
                plt.plot(q2.index, q2, linestyle=diagnoses_lines[D], color=datasets_colors[j], linewidth=2.5)

                # Std
                #plt.fill_between(mean.index, mean - std, mean + std, alpha=0.1, color=datasets_colors[dataset_name])
                #plt.fill_between(q2.index, q1, q3, alpha=0.1, color=datasets_colors[dataset_name])

            # Plot grey shadow for this diagnosis, with global q1 and q3
            all_y = pd.concat([babilony_quality[(region, dataset_name, D)] for dataset_name in datasets_names])
            q1, q2, q3 = all_y.quantile(0.25), all_y.quantile(0.5), all_y.quantile(0.75)
            plt.fill_between(q2.index, q1, q3, alpha=0.1, color=diagnoses_shadow[D])

        plt.title(f"{region}")
        plt.ylabel('eLORETA Current Density')
        xlabels = [get_band_abbrev(band) for band in to_keep]
        plt.xticks(range(len(to_keep)), xlabels)
        plt.title(region)

    plt.subplots_adjust(hspace=0.3, wspace=0.3)
    plt.legend(loc='best', fontsize=6)
    plt.tight_layout()
    run["quality_control/babiloni_spectrum"].upload(File.as_image(plt.gcf()))
    plt.savefig(join(out_path, f"babiloni_quality_control.{image_format}"), dpi=dpi, transparent=True, bbox_inches='tight')
    plt.close()

def plot_mean_diffs(run, _datasets, outpath, log=False):
    # y-axis: difference between the average feature value at site 1 and the average feature value at site 2
    # x-axis: average feature value across all participants from both sites
    # only izmir and newcastle

    ymin, ymax = -6, 2
    if log:
        ymin, ymax = -1, 1

    all_dataset_names = list(_datasets.keys())
    combinations = list(itertools.combinations(all_dataset_names, 2))

    # Each row can have 4 columns. How many rows do we need?
    num_rows = (len(combinations) + 3) // 4

    fig = plt.figure(figsize=(4 * 4, 3 * num_rows))
    for i, (dataset_name_1, dataset_name_2) in enumerate(combinations):
        if dataset_name_1 == dataset_name_2:
            break

        plt.subplot(num_rows, 4, i + 1)

        # Extract datasets for Izmir and Newcastle
        d1_data = _datasets[dataset_name_1]
        d2_data = _datasets[dataset_name_2]

        # Calculate the difference and average
        diff_values = d1_data.mean() - d2_data.mean()
        avg_values = (d1_data.mean() + d2_data.mean()) / 2

        # Plotting
        plt.scatter(avg_values, diff_values, color='blue')
        plt.xlabel('Average feature value across sites')
        plt.ylabel('Feature difference across sites')
        plt.axhline(0, color='red', linestyle='--')
        plt.ylim(ymin, ymax)
        plt.title(f'{dataset_name_1} - {dataset_name_2}')

    plt.subplots_adjust(hspace=0.3, wspace=0.3)
    plt.tight_layout()
    run["quality_control/mean_diffs"].upload(File.as_image(plt.gcf()))
    plt.savefig(join(outpath, f"mean_diffs.{image_format}"), dpi=dpi, transparent=True, bbox_inches='tight')
    plt.close(fig)

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


def plot_qq(run, _datasets, out_path):
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
        run[f"quality_control/qq/{dataset_name}"].upload(File.as_image(plt.gcf()))
        plt.savefig(join(out_path, f"qq_{dataset_name}.{image_format}"), dpi=dpi, transparent=True, bbox_inches='tight')
        plt.close(fig)

def plot_correlation_with_var(run, in_path, out_path, var_of_interest, variant):
    # x-axis: site/batch
    # y-axis: percentage of features significantly correlated with the variable of interest

    # Read correlations csv -> pandas dataframe
    correlations = pd.read_csv(join(in_path, f"correlation_{var_of_interest}.csv"), index_col=0)
    run[f"correlation/{var_of_interest}/all"].upload(join(in_path, f"correlation_{var_of_interest}.csv"))

    # Percentage of features significantly correlated with the variable of interest
    N_significant = sum(correlations['p'] < 0.05)
    significant = N_significant / len(correlations) * 100

    # plot
    plt.figure(figsize=(3, 4))
    plt.bar(variant, significant)
    plt.ylim(0, 100)
    plt.ylabel(f"% of features significantly correlated with {var_of_interest.capitalize()}")
    plt.tight_layout()
    run[f"correlation/{var_of_interest}/significant"].upload(File.as_image(plt.gcf()))
    plt.savefig(join(out_path, f"correlation_with_{var_of_interest}.{image_format}"), dpi=dpi, transparent=True, bbox_inches='tight')
    plt.close()

def plot_classification_with_var(run, in_path, out_path, var_of_interest, variant):
    # x-axis: site/batch
    # y-axis: percentage of features significantly correlated with the variable of interest

    # Read classifications csv -> pandas dataframe
    classification_res = pd.read_csv(join(in_path, f"classification_{var_of_interest}.csv"), index_col=0)
    pred = classification_res['pred']
    true = classification_res['true']

    # Compute classification metrics
    report = classification_report(true, pred, output_dict=True)
    run[f"classification/{var_of_interest}/my_metrics"] = report

    # Compute classification metrics by dataset
    # The indexes can be separated by '-' to get the dataset name
    datasets = set([i.split("-")[0] for i in true.index])
    for dataset in datasets:
        idx = [i for i in true.index if i.startswith(dataset)]
        report_dataset = classification_report(true.loc[idx], pred.loc[idx], output_dict=True)
        run[f"classification/{var_of_interest}/{dataset}"] = report_dataset

    print('W. F1-Score:', report['weighted avg']['f1-score'])

    # Plot weighted F1-Score
    fig = plt.figure(figsize=(3, 4))
    plt.bar(variant, report['weighted avg']['f1-score'])
    # Write value on bar
    plt.text(0, report['weighted avg']['f1-score'], f"{report['weighted avg']['f1-score']:.2f}", ha='center', va='bottom')
    plt.ylim(0, 1.0)
    plt.ylabel(f"F1-score (w.avg) to classify {var_of_interest.capitalize()}")
    plt.tight_layout()
    plt.savefig(join(out_path, f"classification_with_{var_of_interest}.{image_format}"), dpi=dpi, transparent=True, bbox_inches='tight')
    plt.close(fig)

    # Plot Confusion matrix
    fig = plt.figure(figsize=(4, 4))
    sns.heatmap(confusion_matrix(true, pred), annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.tight_layout()
    run[f"classification/{var_of_interest}/confusion_matrix"].upload(File.as_image(plt.gcf()))
    plt.savefig(join(out_path, f"confusion_matrix_{var_of_interest}.{image_format}"), dpi=dpi, transparent=True, bbox_inches='tight')
    plt.close(fig)

    """
    # Plot ROC-AUC curve with sklearn library
    plot_roc_curve(true.to_numpy(), pred.to_numpy())
    run[f"classification/{var_of_interest}/roc_auc"].upload(File.as_image(plt.gcf()))
    plt.savefig(join(out_path, f"roc_auc_{var_of_interest}.{image_format}"), dpi=dpi, transparent=True, bbox_inches='tight')
    auc_score = roc_auc_score(true, pred, sample_weight=sample_weight)
    run[f"classification/{var_of_interest}/roc_auc_score"] = auc_score
    """

def plot_2components(run, _metadata, in_path, out_path, method):
    filepath = join(in_path, f"{method}_transformed.csv")
    if exists(filepath):
        # Read csv principal components file
        pc = pd.read_csv(filepath, index_col=0)

        # Plot stlyles
        dataset_colors = {"Newcastle": "blue", "Izmir": "green", "Istambul": "orange", "Miltiadous": "pink", "BrainLat:AR": "black", "BrainLat:CL": "yellow"}
        #dataset_colors = {l: [c / 255 for c in color] for l, color in dataset_colors.items()}
        diagnoses_circles = {"HC": "x", "AD": "o"}

        # One plot, all datasets
        fig = plt.figure(figsize=(6, 6))
        for dataset, color in dataset_colors.items():
            for diagnosis, marker in diagnoses_circles.items():
                idx = _metadata[(_metadata["DIAGNOSIS"] == diagnosis) & (_metadata["SITE"] == dataset)].index
                existing_idx = pc.index.intersection(idx)
                plt.scatter(pc['0'].loc[existing_idx], pc['1'].loc[existing_idx], color=color,
                            label=f"{dataset} - {diagnosis}", marker=marker)

        plt.legend(loc='best', fontsize=8)
        plt.xlabel("Component 1", fontsize=14)
        plt.ylabel("Component 2", fontsize=14)
        plt.tight_layout()
        run[f"quality_control/{method}"].upload(File.as_image(plt.gcf()))
        plt.savefig(join(out_path, f"{method}.{image_format}"), dpi=dpi, transparent=True, bbox_inches='tight')
        plt.close(fig)

def plot_distance_matrix(run, in_path, out_path):
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
    run["quality_control/distance_matrix"].upload(File.as_image(plt.gcf()))
    plt.savefig(join(out_path, f"distance_matrix.{image_format}"), dpi=dpi, transparent=True, bbox_inches='tight')
    plt.close()


def plot_batch_effects_dist(run, in_path, out_path):
    print("Creating batch effects distribution plots.")
    # Load distributions_by_dataset.pkl
    distributions = pd.read_pickle(join(in_path, "distributions_by_dataset.pkl"))

    # visualize fit of the prior distribution, along with the observed distribution of site effects
    colors = ['blue', 'red', 'green', 'orange', 'pink', 'black', 'yellow']

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
    run["harmonization/additive_batch_effects"].upload(File.as_image(plt.gcf()))
    plt.savefig(join(out_path, f"add_batch_effects_dist.{image_format}"), dpi=dpi, transparent=True, bbox_inches='tight')
    plt.close()

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
    run["harmonization/multiplicative_effects"].upload(File.as_image(plt.gcf()))
    plt.savefig(join(out_path, f"mul_batch_effects_dist.{image_format}"), dpi=dpi, transparent=True, bbox_inches='tight')
    plt.close()


def plot_mmse_distribution(run, datasets_metadata, in_path, out_path):
    # Get dataset names
    dataset_names = datasets_metadata.keys()

    # Get all MMSE data
    datasets_metadata = {k: v for k, v in datasets_metadata.items() if k in dataset_names}
    datasets_metadata = {k: v[['MMSE', 'DIAGNOSIS']] for k, v in datasets_metadata.items()}

    # Get subjects used in the analysis
    subjects = pd.read_csv(join(in_path, "lda_transformed.csv"), index_col=0).index
    # Get all dataset_names from the subjects index and put it in a column
    subjects_datasets = [s.split("-")[0] for s in subjects]
    subjects = pd.DataFrame(subjects_datasets, index=subjects, columns=['Dataset'])

    # Filter MMSE data for the subjects used in the analysis
    for dataset_name in dataset_names:
        these_subjects = subjects[subjects['Dataset'] == dataset_name].index
        filtered_data = datasets_metadata[dataset_name].loc[these_subjects]
        datasets_metadata[dataset_name] = filtered_data

   # Combine the counts for each dataset and diagnosis into a single DataFrame
    combined_counts = pd.DataFrame()

    for dataset_name, mmse in datasets_metadata.items():
        for D in ("HC", "AD"):
            mmse_D = mmse[datasets_metadata[dataset_name]['DIAGNOSIS'] == D]
            mmse_D = mmse_D['MMSE']
            counts = mmse_D.value_counts(normalize=True).sort_index()
            counts_df = counts.reset_index()
            counts_df.columns = ['MMSE', 'Density']
            counts_df['Dataset'] = dataset_name
            counts_df['Diagnosis'] = D
            combined_counts = pd.concat([combined_counts, counts_df], ignore_index=True)

    # Combine the Dataset and Diagnosis into a single column
    combined_counts['Dataset_Diagnosis'] = combined_counts['Dataset'] + '_' + combined_counts['Diagnosis']

    # Plot
    plt.figure(figsize=(10, 2))
    sns.barplot(x='MMSE', y='Density', hue='Dataset_Diagnosis', data=combined_counts)
    plt.legend(loc='upper left', fontsize=5)
    plt.xlabel("MMSE")
    plt.ylabel("Density")
    plt.tight_layout()
    run["datasets/mmse_distribution"].upload(File.as_image(plt.gcf()))
    plt.savefig(join(out_path, f"mmse_distribution.{image_format}"), dpi=dpi, transparent=True, bbox_inches='tight')
    plt.close()


def plot_simple_diagnosis_discriminant(run, datasets_metadata, in_path, out_path):
    # Read csv principal components file
    pc = pd.read_csv(join(in_path, f"simple_diagnosis_discriminant_transformed.csv"), index_col=0)

    # Load model
    model = load(open(join(in_path, "simple_diagnosis_discriminant.pkl"), 'rb'))

    # Plot stlyles
    diagnoses_colors = {"HC": "blue", "AD": "red"}

    # One plot, all datasets
    if len(pc.columns) <= 2:
        fig = plt.figure(figsize=(6, 6))
        for diagnosis, color in diagnoses_colors.items():
            idx = datasets_metadata[datasets_metadata["DIAGNOSIS"] == diagnosis].index
            existing_idx = pc.index.intersection(idx)
            match len(pc.columns):
                case 1:
                    plt.scatter(pc['0'].loc[existing_idx], np.zeros(len(existing_idx)), color=color, label=diagnosis, alpha=0.5)
                    plt.xlabel("Component 1", fontsize=14)
                case 2:
                    plt.scatter(pc['0'].loc[existing_idx], pc['1'].loc[existing_idx], color=color, label=diagnosis, alpha=0.5)
                    plt.xlabel("Component 1", fontsize=14)
                    plt.ylabel("Component 2", fontsize=14)
    else:
        # Number of pair-wise combinations?
        num_combinations = len(list(itertools.combinations(pc.columns, 2)))
        num_cols = 3
        num_rows = (num_combinations + num_cols - 1) // num_cols
        fig = plt.figure(figsize=(4 * num_cols, 3 * num_rows))
        for i, (col1, col2) in enumerate(itertools.combinations(pc.columns, 2)):
            plt.subplot(num_rows, num_cols, i + 1)
            for diagnosis, color in diagnoses_colors.items():
                idx = datasets_metadata[datasets_metadata["DIAGNOSIS"] == diagnosis].index
                existing_idx = pc.index.intersection(idx)
                plt.scatter(pc[col1].loc[existing_idx], pc[col2].loc[existing_idx], color=color, label=diagnosis, alpha=0.5)
                plt.xlabel(f"Component {col1}", fontsize=14)
                plt.ylabel(f"Component {col2}", fontsize=14)

            """
            ax = fig.add_subplot(projection='3d')
            ax.scatter(
                pc.loc[existing_idx, '0'],
                pc.loc[existing_idx, '1'],
                pc.loc[existing_idx, '2'],
                color=color, label=diagnosis
            )
            ax.set_xlabel('Component 1')
            ax.set_ylabel('Component 2')
            ax.set_zlabel('Component 3')
            plt.show(block=True)
            """

    plt.legend(loc='best', fontsize=8)
    plt.tight_layout()
    run[f"simple_diagnosis_discriminant/visual_separation"].upload(File.as_image(plt.gcf()))
    plt.savefig(join(out_path, f"simple_diagnosis_discriminant.{image_format}"), dpi=dpi, transparent=True, bbox_inches='tight')
    plt.close(fig)
