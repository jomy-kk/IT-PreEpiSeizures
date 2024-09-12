from pandas import DataFrame, read_csv


def feature_wise_normalisation(features: DataFrame, method: str = 'mean-std') -> DataFrame:
    """
    Normalise feature matrices in a feature-wise manner.
    The given DataFrame must be in the shape (n_samples, n_features).
    """
    min,max, mean, std = features.min(), features.max(), features.mean(), features.std()
    #coefficients.to_csv('kjpp_coefficients.csv')
    if method == 'mean-std':
        return (features-mean)/std
    elif method == 'min-max':
        return (features-min)/(max-min)
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


diagnoses_groups = {
    # Mental Disorders (usually of adult onset)
    'Psychotic Disorders': ['F06.0', 'F06.2', 'F23', 'F24', 'F28', 'F29'],
    'Schizo and Delusional Disorders': ['F20', 'F21', 'F22', 'F25'],
    'Mood Disorders': ['F06.3', 'F30', 'F31', 'F34', 'F39'],
    'Depressive Disorders': ['F32', 'F33'],
    'Anxiety Disorders': ['F06.4', 'F40', 'F41'],
    'Obsessive-Compulsive Disorders': ['F42'],
    'Stress-related Disorders': ['F43'],
    'Dissociative Disorders': ['F44'],
    'Somatoform Disorders': ['F45'],
    'Cognitive Disorders': ['F06.7', ],
    'Personality and Behaviour Disorders': ['F07', 'F59', 'F60', 'F63'],
    'Mental and Behavioural from Psychoactives': ['F10', 'F11', 'F12', 'F13', 'F14', 'F15', 'F16', 'F17', 'F18', 'F19'],
    'Eating Disorders': ['F50'],
    'Mental Sleep Disorders': ['F51'],
    'Other Mental Disorders': ['F06.8', 'F09', 'F48', 'F54'],

    # Mental Disorders (usually of child onset)
    'Intellectual Disabilities': ['F70', 'F71', 'F72', 'F73', 'F78', 'F79'],
    'Developmental Speech and Language Disorders': ['F80', ],
    'Developmental Scholastic Disorders': ['F81', ],
    'Developmental Motor Disorders': ['F82', ],
    'Developmental Pervasive Disorders': ['F84', ],
    'Attention-deficit Hyperactivity Disorders': ['F90', ],
    'Conduct Disorders': ['F91', ],
    'Emotional Disorders': ['F93', ],
    'Tic disorders': ['F95', ],
    'Other Developmental Disorders': ['F88', 'F89', 'F98'],

    # Neurological Disorders
    'Epilepsies and Status Epilepticus': ['G40', 'G41'],
    'Migranes and Headaches': ['G43', 'G44'],
    'Ischemic and Vascular Brain Syndromes': ['G45', 'G46'],
    'Sleep Disorders': ['G47'],
    'CNS Inflammatory Diseases': ['G01', 'G02', 'G03', 'G04', 'G05', 'G06', 'G07', 'G08', 'G09'],
    'CNS Atrophies': ['G10', 'G11', 'G12', 'G13', 'G14'],
    'Extrapyramidal and Movement Disorders': ['G23', 'G24', 'G25', 'G26'],
    'CNS Demyelinating Diseases': ['G35', 'G36', 'G37', 'G38', 'G40'],
    'Neuropathies and Plexopathies': ['G50', 'G51', 'G52', 'G53', 'G54', 'G55', 'G56', 'G57', 'G58', 'G59', 'G60', 'G61', 'G62', 'G63', 'G64'],
    'Myo(neuro)pathies': ['G70', 'G71', 'G72', 'G73'],
    'Cerebral Palsy and Paralytic Syndromes': ['G80', 'G81', 'G82', 'G83'],
    'Other CNS or PNS Disorders': ['G90', 'G91', 'G92', 'G93', 'G94', 'G95', 'G96', 'G97', 'G98', 'G99'],

    # Other groups
    'Nutrition and Metabolic Disorders': ['E' + str(i).zfill(2) for i in range(40, 91)],
    'Endocrine Disorders': ['E' + str(i).zfill(2) for i in range(0, 40)],
    'Liver Diseases': ['K70', 'K71', 'K72', 'K73', 'K74', 'K75', 'K76', 'K77'],
    'Congenital Nervous Malformations': ['Q00', 'Q01', 'Q02', 'Q03', 'Q04', 'Q05', 'Q06', 'Q07'],
    'Chromosomal Abnormalities': ['Q90', 'Q91', 'Q92', 'Q93', 'Q95', 'Q96', 'Q97', 'Q98', 'Q99'],
}

