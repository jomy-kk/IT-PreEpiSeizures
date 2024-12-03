import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from scipy.stats import norm, invgamma

from neuroCombat import neuroCombat
from neuroHarmonize import harmonizationLearn
from pycombat import Combat


def _check_input(_X: pd.DataFrame, _metadata: pd.DataFrame):
    assert _X.index.tolist() == _metadata.index.tolist()
    data = _X.to_numpy(dtype=np.float32)
    covariates: pd.DataFrame = _metadata.copy()
    return data, covariates


def _prepare_covariates(covariates: pd.DataFrame, cov_gender=True, cov_age=True, cov_education=True):
    # keep only the age and gender columns
    covariates = covariates[['AGE', 'GENDER', 'EDUCATION YEARS', 'SITE']]
    # binarize gender: M=1, F=0
    covariates['GENDER'].replace('M', 1, inplace=True)
    covariates['GENDER'].replace('F', 0, inplace=True)

    # test
    if not cov_gender:
        covariates.drop(columns='GENDER', inplace=True)
    if not cov_age:
        covariates.drop(columns='AGE', inplace=True)
    if not cov_education:
        covariates.drop(columns='EDUCATION YEARS', inplace=True)

    print("Covariates:", covariates.columns)
    return covariates





def original_combat(_X: pd.DataFrame, _metadata: pd.DataFrame,
                    cov_gender=True, cov_age=True, cov_education=True) -> pd.DataFrame:
    data, covariates = _check_input(_X, _metadata)
    covariates = _prepare_covariates(covariates, cov_gender, cov_age, cov_education)

    biological_covars = covariates.copy()
    biological_covars.drop(columns=['SITE'], inplace=True)
    biological_covars = biological_covars.to_numpy(dtype=np.float32)
    batches = covariates['SITE'].to_numpy()

    # Harmonization step
    combat_model = Combat()
    harmonized_data = combat_model.fit_transform(Y=data, b=batches, X=biological_covars)

    # Batch effects estimates
    """
    sites = combat_model.batches_
    gamma_hat, delta_hat = [], []
    gamma_bar, t2 = [], []
    a_prior, b_prior = [], []
    for i, dataset_name in enumerate(sites):
        gamma_bar.append(combat_model.gamma_[i])
        t2.append(combat_model.delta_sq_['t2'][i])
        gamma_hat.append(estimates['gamma_hat'][i])
        a_prior.append(estimates['a_prior'][i])
        b_prior.append(estimates['b_prior'][i])
        delta_hat.append(estimates['delta_hat'][i])
    _plot_estimates(sites, gamma_bar, t2, gamma_hat, a_prior, b_prior, delta_hat)
    """

    return pd.DataFrame(harmonized_data, index=_X.index, columns=_X.columns)


def neuro_combat(_X: pd.DataFrame, _metadata: pd.DataFrame,
                 cov_gender=True, cov_age=True, cov_education=True) -> tuple[pd.DataFrame, dict]:

    data, covariates = _check_input(_X, _metadata)
    data = data.T  # data should be (features, samples)
    covariates = _prepare_covariates(covariates, cov_gender, cov_age, cov_education)

    # To specify the name of the variable that encodes for the batch covariate
    batch_col = 'SITE'
    # To specify names of the variables that are categorical:
    categorical_cols = ['GENDER', ]
    continuous_cols = ['AGE', 'EDUCATION YEARS']

    # Harmonization step
    res = neuroCombat(dat=data, covars=covariates, batch_col=batch_col, categorical_cols=categorical_cols, continuous_cols=continuous_cols)
    harmonized_data, estimates = res['data'].T, res['estimates']

    # Batch effects estimates
    sites = estimates['batches']
    gamma_hat, delta_hat = [], []
    gamma_bar, t2 = [], []
    a_prior, b_prior = [], []
    for i, dataset_name in enumerate(sites):
        gamma_bar.append(estimates['gamma_bar'][i])
        t2.append(estimates['t2'][i])
        gamma_hat.append(estimates['gamma_hat'][i])
        a_prior.append(estimates['a_prior'][i])
        b_prior.append(estimates['b_prior'][i])
        delta_hat.append(estimates['delta_hat'][i])
    dist_parameters = {'sites': sites, 'gamma_bar': gamma_bar, 't2': t2, 'gamma_hat': gamma_hat, 'a_prior': a_prior,
                       'b_prior': b_prior, 'delta_hat': delta_hat}

    return pd.DataFrame(harmonized_data, index=_X.index, columns=_X.columns), dist_parameters


def neuro_harmonize(_X: pd.DataFrame, _metadata: pd.DataFrame,
                    cov_gender=True, cov_age=True, cov_education=True, cov_diagnosis=False) -> tuple[pd.DataFrame, dict]:

    assert _X.index.tolist() == _metadata.index.tolist()
    covariates: pd.DataFrame = _metadata.copy()
    # keep only the age and gender columns
    covariates = covariates[['AGE', 'GENDER', 'EDUCATION YEARS', 'SITE', 'DIAGNOSIS']]
    # binarize gender: M=1, F=0
    covariates['GENDER'].replace('M', 1, inplace=True)
    covariates['GENDER'].replace('F', 0, inplace=True)
    # binarize gender: M=1, F=0
    covariates['DIAGNOSIS'].replace('AD', 1, inplace=True)
    covariates['DIAGNOSIS'].replace('HC', 0, inplace=True)

    # test
    if not cov_gender:
        covariates.drop(columns='GENDER', inplace=True)
    if not cov_age:
        covariates.drop(columns='AGE', inplace=True)
    if not cov_education:
        covariates.drop(columns='EDUCATION YEARS', inplace=True)
    if not cov_diagnosis:
        covariates.drop(columns='DIAGNOSIS', inplace=True)

    print("Covariates:", covariates.columns)

    # Drop rows with missing values in covariates
    covariates.dropna(inplace=True, axis=0)
    _X = _X.loc[covariates.index]
    data = _X.to_numpy(dtype=np.float32)

    # run
    combat_model, harmonized_data = harmonizationLearn(data, covariates)

    # Batch effects estimates
    sites = combat_model['SITE_labels']
    gamma_hat, delta_hat = [], []
    gamma_bar, t2 = [], []
    a_prior, b_prior = [], []
    for i, dataset_name in enumerate(sites):
        gamma_bar.append(combat_model['gamma_bar'][i])
        t2.append(combat_model['t2'][i])
        gamma_hat.append(combat_model['gamma_hat'][i, :])
        a_prior.append(combat_model['a_prior'][i])
        b_prior.append(combat_model['b_prior'][i])
        delta_hat.append(combat_model['delta_hat'][i, :])
    dist_parameters = {'sites': sites, 'gamma_bar': gamma_bar, 't2': t2, 'gamma_hat': gamma_hat, 'a_prior': a_prior, 'b_prior': b_prior, 'delta_hat': delta_hat}
    #_plot_estimates(sites, gamma_bar, t2, gamma_hat, a_prior, b_prior, delta_hat)

    return pd.DataFrame(harmonized_data, index=_X.index, columns=_X.columns), dist_parameters


def opnested_combat(_X: pd.DataFrame, _metadata: pd.DataFrame,
                 cov_gender=True, cov_age=True, cov_education=True) -> pd.DataFrame:

    data, covariates = _check_input(_X, _metadata)
    data = data.T  # data should be (features, samples)
    covariates = _prepare_covariates(covariates, cov_gender, cov_age, cov_education)

    # To specify the name of the variable that encodes for the batch covariate
    batch_list = ['SCANNER', 'SITE', ]
    # To specify names of the variables that are categorical:
    categorical_cols = ['GENDER', ]
    continuous_cols = ['AGE', 'EDUCATION YEARS']

    # Add column 'SCANNER' to the covariates, where 'Izmir' and 'Newcastle' have the same scanner
    covariates['SCANNER'] = covariates['SITE'].replace({'Izmir': 'Scanner1', 'Newcastle': 'Scanner1', 'Sapienza': 'Scanner2'})

    # Harmonization step
    # # FOR GMM COMBAT VARIANTS:
    # # Adding GMM Split to batch effects
    gmm_df = nested.GMMSplit(dat, caseno, filepath2)
    gmm_df_merge = covars_df.merge(gmm_df, right_on='Patient', left_on='Unnamed: 0')
    covars['GMM'] = gmm_df_merge['Grouping']

    # # EXECUTING OPNESTED+GMM COMBAT
    # # Here we add the newly generated GMM grouping to the list of batch variables for harmonization
    # batch_list = batch_list + ['GMM']

    # EXECUTING OPNESTED-GMM COMBAT
    # Here we add the newly generated GMM grouping to the list of categorical variables that will be protected during
    # harmonization
    categorical_cols = categorical_cols + ['GMM']

    # Completing Nested ComBat
    output_df = nested.OPNestedComBat(dat, covars, batch_list, filepath2, categorical_cols=categorical_cols,
                                      continuous_cols=continuous_cols)

    # Batch effects estimates
    sites = estimates['batches']
    gamma_hat, delta_hat = [], []
    gamma_bar, t2 = [], []
    a_prior, b_prior = [], []
    for i, dataset_name in enumerate(sites):
        gamma_bar.append(estimates['gamma_bar'][i])
        t2.append(estimates['t2'][i])
        gamma_hat.append(estimates['gamma_hat'][i])
        a_prior.append(estimates['a_prior'][i])
        b_prior.append(estimates['b_prior'][i])
        delta_hat.append(estimates['delta_hat'][i])
    _plot_estimates(sites, gamma_bar, t2, gamma_hat, a_prior, b_prior, delta_hat)

    return pd.DataFrame(harmonized_data, index=_X.index, columns=_X.columns)

