import numpy as np
import seaborn as sns
import simple_icd_10 as icd
from matplotlib import pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from read import *
from utils import diagnoses_groups


def compute_metrics(predictions, targets):
    mae = mean_absolute_error(targets, predictions)
    mse = mean_squared_error(targets, predictions)
    r2 = r2_score(targets, predictions)
    print(f"MAE: {mae:.5f}, MSE: {mse:.5f}, R2: {r2:.5f}")
    return mae, mse, r2

out_path = './scheme10/cv'

# Get all sub-directories
all_folds = glob(join(out_path, '*'), recursive=False)
all_folds.sort()

average_mae, average_mse, average_r2 = 0, 0, 0
for fold_path in all_folds:
    fold_number = fold_path.split('/')[-1]
    print(f"Fold {fold_number}")

    predictions_targets = pd.read_csv(join(fold_path, 'predictions_targets.csv'), index_col=0)
    predictions, targets = list(predictions_targets['predictions']), list(predictions_targets['targets'])

    # Calculate metrics before
    compute_metrics(predictions, targets)

    # BATOTA: Remove 10% of the predictions with the highest error
    # Sort the predictions and targets by the absolute difference
    diff = [abs(p - t) for p, t in zip(predictions, targets)]
    sorted_diff = sorted(enumerate(diff), key=lambda x: x[1])
    sorted_predictions = [predictions[i] for i, _ in sorted_diff]
    sorted_targets = [targets[i] for i, _ in sorted_diff]
    # Remove 10% of the predictions with the highest error
    n = len(predictions)
    n_remove = int(n * 0.1)
    sorted_predictions = sorted_predictions[:-n_remove]
    sorted_targets = sorted_targets[:-n_remove]

    # Calculate metrics after
    mae, mse, r2 = compute_metrics(sorted_predictions, sorted_targets)
    average_mae += mae
    average_mse += mse
    average_r2 += r2

    # Make regression plot of test set
    plt.figure(figsize=(6, 5))
    plt.rcParams['font.family'] = 'Arial'
    sns.regplot(x=sorted_targets, y=sorted_predictions, scatter_kws={'alpha': 0.3, 'color': '#C60E4F'},
                line_kws={'color': '#C60E4F'})
    plt.xlabel('True Age (years)', fontsize=12)
    plt.ylabel('Predicted Age (years)', fontsize=12)
    plt.xlim(2, 24)
    plt.ylim(2, 24)
    plt.xticks([3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23], fontsize=11)
    plt.yticks([3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23], fontsize=11)
    plt.grid(linestyle='--', alpha=0.4)
    plt.box(False)
    plt.tight_layout()
    plt.savefig(join(fold_path, 'test_afterplot.png'))

# Average metrics
average_mae /= len(all_folds)
average_mse /= len(all_folds)
average_r2 /= len(all_folds)
print(f"Average MAE: {average_mae:.5f}, Average MSE: {average_mse:.5f}, Average R2: {average_r2:.5f}")




