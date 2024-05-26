from pandas import DataFrame, read_csv


def feature_wise_normalisation(features: DataFrame, method: str = 'mean-std') -> DataFrame:
    """
    Normalise feature matrices in a feature-wise manner.
    The given DataFrame must be in the shape (n_samples, n_features).
    """
    coefficients = DataFrame([features.min(), features.max(), features.mean(), features.std()], index=['min', 'max', 'mean', 'std'])
    #coefficients.to_csv('kjpp_coefficients.csv')
    if method == 'mean-std':
        return (features-coefficients.loc['mean'])/coefficients.loc['std']
    elif method == 'min-max':
        return (features-coefficients.loc['min'])/(coefficients.loc['max']-coefficients.loc['min'])
    else:
        raise ValueError("Invalid method. Choose from 'mean-std' or 'min-max'.")


def feature_wise_normalisation_with_coeffs(features: DataFrame, method: str, coefficients_filepath: str):
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

