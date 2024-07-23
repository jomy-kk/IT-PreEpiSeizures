import numpy as np
from pandas import DataFrame, read_csv
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


ELDERS_COLOUR = '#C60E4F'
CHILDREN_COLOUR = '#0067B1'

def feature_wise_normalisation(features: DataFrame, method: str = 'mean-std', save=None) -> DataFrame:
    """
    Normalise feature matrices in a feature-wise manner.
    The given DataFrame must be in the shape (n_samples, n_features).
    """
    coefficients = DataFrame([features.min(), features.max(), features.mean(), features.std()], index=['min', 'max', 'mean', 'std'])
    if save is not None:
        coefficients.to_csv(save)
    if method == 'mean-std':
        return (features-coefficients.loc['mean'])/coefficients.loc['std']
    elif method == 'min-max':
        return (features-coefficients.loc['min'])/(coefficients.loc['max']-coefficients.loc['min'])
    else:
        raise ValueError("Invalid method. Choose from 'mean-std' or 'min-max'.")


def feature_wise_normalisation_with_coeffs(features: DataFrame, method: str, coefficients_filepath: str) -> DataFrame:
    """
    Normalise feature matrices in a feature-wise manner.
    The given DataFrame must be in the shape (n_samples, n_features).
    """
    coefficients = read_csv(coefficients_filepath, index_col=0)
    if method == 'mean-std':
        return (features-coefficients.loc['mean'])/coefficients.loc['std']
    elif method == 'min-max':
        return (features-coefficients.loc['min'])/(coefficients.loc['max']-coefficients.loc['min'])
    else:
        raise ValueError("Invalid method. Choose from 'mean-std' or 'min-max'.")

def weighted_error(predictions, targets, targets_int=True) -> tuple[float, float, float]:
    """
    Returns the weighted mean absolute and squared error and R2 between predictions and targets.
    """

    if not targets_int:
        targets_classes = np.round(targets).astype(int)
    else:
        targets_classes = targets

    # Class frequencies
    unique_classes, class_counts = np.unique(targets_classes, return_counts=True)
    class_frequencies = dict(zip(unique_classes, class_counts))

    # Inverse of class frequencies
    class_weights = {cls: 1.0 / freq for cls, freq in class_frequencies.items()}

    # Assign weights to samples
    sample_weights = np.array([class_weights[cls] for cls in targets_classes])

    # Compute weighted metrics
    weighted_mse = mean_squared_error(targets, predictions, sample_weight=sample_weights)
    weighted_mae = mean_absolute_error(targets, predictions, sample_weight=sample_weights)
    weighted_r2 = r2_score(targets, predictions, sample_weight=sample_weights)

    return weighted_mae, weighted_mse, weighted_r2
