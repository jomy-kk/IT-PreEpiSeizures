import re

import numpy as np
from pandas import DataFrame, read_csv
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def feature_wise_normalisation(features: DataFrame, method: str = 'mean-std') -> DataFrame:
    """
    Normalise feature matrices in a feature-wise manner.
    The given DataFrame must be in the shape (n_samples, n_features).
    """
    coefficients = DataFrame([features.min(), features.max(), features.mean(), features.std()], index=['min', 'max', 'mean', 'std'])
    if method == 'mean-std':
        return (features-coefficients.loc['mean'])/coefficients.loc['std']
    elif method == 'min-max':
        return (features-coefficients.loc['min'])/(coefficients.loc['max']-coefficients.loc['min'])
    else:
        raise ValueError("Invalid method. Choose from 'mean-std' or 'min-max'.")


def feature_wise_normalisation_with_coeffs(features: DataFrame, method: str, coefficients: DataFrame) -> DataFrame:
    """
    Normalise feature matrices in a feature-wise manner.
    The given DataFrame must be in the shape (n_samples, n_features).
    """
    if method == 'mean-std':
        return (features-coefficients.loc['mean'])/coefficients.loc['std']
    elif method == 'min-max':
        return (features-coefficients.loc['min'])/(coefficients.loc['max']-coefficients.loc['min'])
    else:
        raise ValueError("Invalid method. Choose from 'mean-std' or 'min-max'.")

def curate_feature_names(F):
    for i, name in enumerate(F):
        components = name.split('#')
        if len(components) == 3:
            if '-' not in components[1]:  # Hjorth
                F[i] = f"{components[0]} {components[1]} {components[2]}"
            else:  # Connectivity
                regions = components[1].replace('-', ' - ')
                F[i] = f"{components[0]} {regions} {components[2].title()}"
        elif len(components) == 4:  # Spectral
            type_ = ' '.join(
                re.findall(r'[A-Z][a-z]*', components[1]))  # convert components[1] from camel case to spaces
            F[i] = f"{type_} {components[2]} {components[3].title()}"

    return F