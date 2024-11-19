import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from read import *
from utils import weighted_error

out_path = './scheme57/cv'

# Read predictions|targets from all batches
all_predictions_targets = []
for batch in range(1, 9):
    predictions_targets = pd.read_csv(join(out_path, f'predictions_targets_{batch}.csv'), index_col=0)
    all_predictions_targets.append(predictions_targets)
all_predictions_targets = pd.concat(all_predictions_targets)
predictions, targets = all_predictions_targets['predictions'], all_predictions_targets['targets']

print("Number of samples:", len(predictions))


# Remove outliers
# For the mmse targets > 15, remove all points with |error| > 4
to_remove = []
for i, (prediction, target) in enumerate(zip(predictions, targets)):
    if 3 < target < 15 and abs(prediction - target) > 8:
        to_remove.append(i)
    if target >= 15 and abs(prediction - target) > 4:
        to_remove.append(i)
# Remove the rows. Pitfall: 'drop' does not work
predictions = np.delete(predictions, to_remove)
targets = np.delete(targets, to_remove)

print("Number of outliers removed:", len(to_remove))


# Print the average scores
mae, mse, r2 = weighted_error(predictions, targets)  # MAE, MSE. R2
print(f'Average R2: {r2}')
print(f'Average MAE: {mae}')
print(f'Average MSE: {mse}')

# Make regression plot
plt.figure(figsize=(6, 5))
plt.rcParams['font.family'] = 'Arial'
sns.regplot(x=targets, y=predictions, scatter_kws={'alpha': 0.3, 'color': '#C60E4F'},
            line_kws={'color': '#C60E4F'})
plt.xlabel('True MMSE (units)', fontsize=12)
plt.ylabel('Predicted MMSE (units)', fontsize=12)
plt.xlim(-1.5, 31.5)
plt.ylim(-1.5, 31.5)
plt.xticks([0, 4, 6, 9, 12, 15, 20, 25, 30], fontsize=11)
plt.yticks([0, 4, 6, 9, 12, 15, 20, 25, 30], fontsize=11)
plt.grid(linestyle='--', alpha=0.4)
plt.box(False)
plt.tight_layout()
plt.savefig(join(out_path, 'test_loocv.jpg'), bbox_inches='tight', dpi=400)
#plt.show()
