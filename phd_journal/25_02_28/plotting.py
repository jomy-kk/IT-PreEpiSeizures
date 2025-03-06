import itertools
from os.path import join, exists
from pickle import load

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from neptune.types import File
from sklearn.metrics import classification_report, confusion_matrix

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
