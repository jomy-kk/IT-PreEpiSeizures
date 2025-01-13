import os

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from neuroharmony import Neuroharmony
from scipy.stats import norm, invgamma

from neuroCombat import neuroCombat
from neuroHarmonize import harmonizationLearn
from pycombat import Combat
from reComBat import reComBat
from sklearn.preprocessing import LabelEncoder

import OPNestedComBat as nested


def _check_input(_X: pd.DataFrame, _metadata: pd.DataFrame):
    assert _X.index.tolist() == _metadata.index.tolist()
    data = _X.to_numpy(dtype=np.float32)
    covariates: pd.DataFrame = _metadata.copy()
    return data, covariates


def _prepare_covariates(covariates: pd.DataFrame, cov_gender=True, cov_age=True, cov_education=True, cov_diagnosis=True):
    columns_to_keep = ['SITE', ]
    if cov_gender:
        columns_to_keep.append('GENDER')
    if cov_age:
        columns_to_keep.append('AGE')
    if cov_education:
        columns_to_keep.append('EDUCATION YEARS')
    if cov_diagnosis:
        columns_to_keep.append('DIAGNOSIS')

    # keep only the age and gender columns
    covariates = covariates[columns_to_keep]
    # binarize gender: M=1, F=0
    covariates['GENDER'].replace('M', 1, inplace=True)
    covariates['GENDER'].replace('F', 0, inplace=True)

    print("Covariates:", covariates.columns)
    return covariates


def original_combat(_X: pd.DataFrame, _metadata: pd.DataFrame,
                    cov_gender=True, cov_age=True, cov_education=True, cov_diagnosis=True) -> pd.DataFrame:
    print("Applying original Combat...")

    data, covariates = _check_input(_X, _metadata)
    covariates = _prepare_covariates(covariates, cov_gender, cov_age, cov_education, cov_diagnosis)

    biological_covars = covariates.copy()
    biological_covars.drop(columns=['SITE'], inplace=True)
    if cov_diagnosis:  # factorize diagnosis
        biological_covars['DIAGNOSIS'] = pd.factorize(biological_covars['DIAGNOSIS'])[0]
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
                 cov_gender=True, cov_age=True, cov_education=True, cov_diagnosis=True) -> tuple[pd.DataFrame, dict]:
    print("Applying NeuroCombat...")
    data, covariates = _check_input(_X, _metadata)
    data = data.T  # data should be (features, samples)
    covariates = _prepare_covariates(covariates, cov_gender, cov_age, cov_education)

    # To specify the name of the variable that encodes for the batch covariate
    batch_col = 'SITE'
    # To specify names of the variables that are categorical:
    categorical_cols = []
    if cov_gender:
        categorical_cols.append('GENDER')
    if cov_diagnosis:
        categorical_cols.append('DIAGNOSIS')
    continuous_cols = []
    if cov_age:
        continuous_cols.append('AGE')
    if cov_education:
        continuous_cols.append('EDUCATION YEARS')

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

    print("Applying NeuroHarmonize...")
    assert _X.index.tolist() == _metadata.index.tolist()
    covariates = _prepare_covariates(_metadata, cov_gender, cov_age, cov_education, cov_diagnosis)
    covariates['DIAGNOSIS'].replace('AD', 1, inplace=True)
    covariates['DIAGNOSIS'].replace('HC', 0, inplace=True)

    # Drop rows with missing values in covariates
    covariates.dropna(inplace=True, axis=0)
    _X = _X.loc[covariates.index]
    data = _X.to_numpy(dtype=np.float32)

    # run
    non_linear_covars = covariates.columns
    non_linear_covars = list(non_linear_covars.drop('SITE'))
    combat_model, harmonized_data = harmonizationLearn(data, covariates, smooth_terms=non_linear_covars)

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


def recombat(_X: pd.DataFrame, _metadata: pd.DataFrame, cov_gender=True, cov_age=True, cov_education=True, cov_diagnosis=False) -> tuple[pd.DataFrame, dict]:
    print("Applying reCombat...")
    assert _X.index.tolist() == _metadata.index.tolist()
    covariates = _prepare_covariates(_metadata, cov_gender, cov_age, cov_education, cov_diagnosis)

    # Drop rows with missing values in covariates
    covariates.dropna(inplace=True, axis=0)
    data = _X.loc[covariates.index]

    biological_covars = covariates.copy()
    biological_covars.drop(columns=['SITE'], inplace=True)
    if cov_diagnosis:  # factorize diagnosis
        biological_covars['DIAGNOSIS'] = pd.factorize(biological_covars['DIAGNOSIS'])[0]
    biological_covars.rename(columns={'AGE': 'AGE_numerical'}, inplace=True)
    batches = covariates['SITE']

    # run
    combat_model = reComBat()
    harmonized_data = combat_model.fit_transform(data, batches, X=biological_covars)

    # Batch effects estimates
    """
    sites = ???
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
    dist_parameters = {'sites': sites, 'gamma_bar': gamma_bar, 't2': t2, 'gamma_hat': gamma_hat, 'a_prior': a_prior,
                       'b_prior': b_prior, 'delta_hat': delta_hat}
    # _plot_estimates(sites, gamma_bar, t2, gamma_hat, a_prior, b_prior, delta_hat)
    """
    return pd.DataFrame(harmonized_data, index=_X.index, columns=_X.columns), None#dist_parameters


def nested_combat(_X: pd.DataFrame, _metadata: pd.DataFrame, gmm_type:str, _out_path: str, cov_gender=True, cov_age=True, cov_education=True, cov_diagnosis=False) -> tuple[pd.DataFrame, dict]:
    print("Applying Nested Combat...")
    assert _X.index.tolist() == _metadata.index.tolist()
    _metadata = _prepare_covariates(_metadata, cov_gender, cov_age, cov_education, cov_diagnosis)

    # Loading in features
    filepath = f"./{_out_path}/GMM/"
    filepath2 = f'./{_out_path}/GMM/'
    if not os.path.exists(filepath2):
        os.makedirs(filepath2)

    datasets = ['Izmir', 'Newcastle', 'Miltiadous', 'Istambul', 'BrainLat:CL', 'BrainLat:AR']

    # Loading in clinical covariates
    covars_df = _metadata
    covars_df['Unnamed: 0'] = covars_df.index
    categorical_cols = ['GENDER', 'DIAGNOSIS', ]
    continuous_cols = ['AGE']

    # Loading in batch effects from dict
    instrumentation = {
        'Newcastle': {'amplifier': 'ANT', 'analog_lowpass': 250, 'sampling': 1024, 'ground': 'R.clav.'},
        'Izmir': {'amplifier': 'BrainAmp', 'analog_lowpass': 70, 'sampling': 500, 'ground': 'R.earlobe'},
        'Istambul': {'amplifier': 'BrainAmp', 'analog_lowpass': 250, 'sampling': 500, 'ground': 'R.earlobe'},
        'Miltiadous': {'amplifier': 'Nihon', 'analog_lowpass': 70, 'sampling': 500, 'ground': 'A1,A2'},
        'BrainLat:AR': {'amplifier': 'Biosemi', 'analog_lowpass': 100, 'sampling': 500, 'ground': 'A1,A2'},
        'BrainLat:CL': {'amplifier': 'Biosemi', 'analog_lowpass': 100, 'sampling': 500, 'ground': 'A1,A2'}
    }
    batch_df = covars_df['SITE'].map(instrumentation).apply(pd.Series)
    batch_list = ['amplifier', 'site']
    batch_df['clc'] = batch_df.index
    batch_df['site'] = covars_df['SITE']

    # Drop unnecessary columns and reset index
    batch_df = batch_df[['clc'] + batch_list]
    batch_df = batch_df.reset_index(drop=True)
    covars_df = covars_df[['Unnamed: 0'] + categorical_cols + continuous_cols]
    covars_df.reset_index(drop=True, inplace=True)

    # CAPTK
    data_df = _X
    data_df['SubjectID'] = data_df.index
    data_df = data_df.reset_index(drop=True)
    data_df = data_df.dropna()
    data_df = data_df.rename(columns={"SubjectID": "Case"})
    data_df = data_df.merge(batch_df['clc'], left_on='Case', right_on='clc')
    caseno_og = data_df['Case'].copy()
    data_df['Case'] = data_df['Case'].str.upper()
    covars_df['Unnamed: 0'] = covars_df['Unnamed: 0'].str.upper()
    batch_df['clc'] = batch_df['clc'].str.upper()
    caseno = data_df['Case']
    dat = data_df.drop(columns=['Case', 'clc'])
    dat = dat.T.apply(pd.to_numeric)

    # Merging batch effects, clinical covariates
    batch_df = data_df[['Case']].merge(batch_df, left_on='Case', right_on='clc')
    covars_df = data_df[['Case']].merge(covars_df, left_on='Case', right_on='Unnamed: 0')
    covars_string = pd.DataFrame()
    covars_string[categorical_cols] = covars_df[categorical_cols].copy()
    covars_string[batch_list] = batch_df[batch_list].copy()
    covars_quant = covars_df[continuous_cols]

    # Encoding categorical variables
    covars_cat = pd.DataFrame()
    for col in covars_string:
        stringcol = covars_string[col]
        le = LabelEncoder()
        le.fit(list(stringcol))
        covars_cat[col] = le.transform(stringcol)

    covars = pd.concat([covars_cat, covars_quant], axis=1)

    # # FOR GMM COMBAT VARIANTS:
    # # Adding GMM Split to batch effects
    gmm_df = nested.GMMSplit(dat, caseno, filepath2)
    gmm_df_merge = covars_df.merge(gmm_df, right_on='Patient', left_on='Unnamed: 0')
    covars['GMM'] = gmm_df_merge['Grouping']

    # # EXECUTING OPNESTED+GMM COMBAT
    # # Here we add the newly generated GMM grouping to the list of batch variables for harmonization
    if gmm_type == '+':
        batch_list = batch_list + ['GMM']

    # EXECUTING OPNESTED-GMM COMBAT
    # Here we add the newly generated GMM grouping to the list of categorical variables that will be protected during
    # harmonization
    if gmm_type == '-':
        categorical_cols = categorical_cols + ['GMM']

    # Completing Nested ComBat
    output_df = nested.OPNestedComBat(dat, covars, batch_list, filepath2, categorical_cols=categorical_cols,
                                      continuous_cols=continuous_cols)
    harmonized_data = output_df
    harmonized_data.index = caseno_og

    # Read GMMorder.txt with numpy
    gmm_order = np.loadtxt(f'{filepath2}/order.txt', dtype=str)

    return harmonized_data, None, gmm_order
