import pandas as pd
from imblearn.over_sampling import SMOTE
from pandas import DataFrame, read_csv
from sklearn.model_selection import StratifiedKFold
import simple_icd_10 as icd

from read import read_diagnoses

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


regions = {
    'Frontal(L)': ('F3', 'F7', 'Fp1'),
    'Frontal(R)': ('F4', 'F8', 'Fp2'),
    'Temporal(L)': ('T3', 'T5'),
    'Temporal(R)': ('T4', 'T6'),
    'Parietal(L)': ('C3', 'P3', ),
    'Parietal(R)': ('C4', 'P4', ),
    'Occipital(L)': ('O2', ),
    'Occipital(R)': ('O1', ),
}

def average_by_region(features: DataFrame):
    """
    Average the features by region.
    """
    new_features = pd.DataFrame()
    region_averaged_features = {}
    for feature in features.columns:
        if 'COH' in feature or 'PLI' in feature:
            # add them as they are in the new DataFrame
            new_features[feature] = features[feature]
        else:
            # Spectral and Hjorth features need to be averaged by region
            for region, channels in regions.items():
                for channel in channels:
                    if channel in feature:
                        if region not in region_averaged_features:
                            region_averaged_features[region] = []
                        region_averaged_features[region].append(feature)
                        break
            # Average the features by region
    for region, features in region_averaged_features.items():
        new_features[region] = features.mean(axis=1)

def augment(features: pd.DataFrame, targets):
    #targets = targets.round().astype(int)

    # Histogram before
    #plt.hist(targets, bins=4, rwidth=0.8)
    #plt.title("Before")
    #plt.show()

    # 4.1. Data Augmentation for regression (target is continuous values)
    """
    features['target'] = targets
    # make indexes sequential numbers
    features.reset_index()
    features, targets = iblr.adasyn(features, 'target', k=3)
    targets = features['target']
    features = features.drop('target', axis=1)
    """

    # 4.2. Data Augmentation method = SMOTE-C

    for k in (5, 4, 3, 2, 1):
        try:
            smote = SMOTE(random_state=42, k_neighbors=k, sampling_strategy='auto')
            features, targets = smote.fit_resample(features, targets)
            print(f"Worked with k={k}")
            break
        except ValueError as e:
            print(f"Did not work with k={k}:", e)


    # Histogram after
    #plt.hist(targets, bins=4, rwidth=0.8)
    #plt.title("After")
    #plt.show()

    # Normalisation after DA
    #features = feature_wise_normalisation(features, method='min-max')
    print("Features shape after DA:", features.shape)

    return features, targets


def custom_cv(objects, targets, n_splits=5, random_state=42):
    """
    Custom Cross-Validation with Data Augmentation on-the-fly.

    Args:
        objects: A DataFrame of feature vectors
        targets: A Series of target values
        n_splits: Number of folds in CV

    Returns:
        The augmented training objects, test objects, training targets, and test targets.
    """

    # Stratify by class
    sss = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    #sss = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    # Select test_size % of examples in a random and stratified way
    split_res = sss.split(objects, targets)

    # for each fold...
    for train_indices, test_indices in split_res:
        print("Train:", len(train_indices))
        print("Test:", len(test_indices))

        # Make sets
        train_objets = objects.iloc[train_indices]
        train_targets = targets.iloc[train_indices]
        test_objects = objects.iloc[test_indices]
        test_targets = targets.iloc[test_indices]

        # Augment train set
        print("Train examples before augmentation:", len(train_targets))
        train_objets, train_targets = augment(train_objets, train_targets)
        print("Train examples after augmentation:", len(train_targets))

        yield train_objets, test_objects, train_targets, test_targets



# make a colour for each diagnoses supergroup
diagnoses_supergroups = ((0, 15), (15, 25), (25, 37), (37, 41))
diagnoses_supergroups_colors = ['red', 'blue', 'green', 'orange']
no_diagnosis_color = 'gray'


def get_classes(subject_codes):

    # Statistics counters
    N_no_list, N_empty_list, N_not_found = 0, 0, 0
    not_found = []


    # Read diagnoses
    diagnoses = read_diagnoses('KJPP')
    colors = []
    for code in subject_codes:
        code = code.split('$')[0]
        subject_diagnoses = diagnoses[code]
        class_decided = None
        if isinstance(subject_diagnoses, list):
            for d in subject_diagnoses:
                if icd.is_valid_item(d):
                    for i, D in enumerate(diagnoses_groups.values()):
                        for el_D in D:
                            if icd.is_valid_item(el_D) and icd.is_descendant(d, el_D):  # diagnosis belongs to this group
                                for j in range(len(diagnoses_supergroups)):
                                    if diagnoses_supergroups[j][0] < i < diagnoses_supergroups[j][1]:
                                        class_decided = diagnoses_supergroups_colors[j]
                                        break
                                break

        colors.append(class_decided)

        if not isinstance(subject_diagnoses, list):
            N_no_list += 1
        elif len(subject_diagnoses) == 0:
            N_empty_list += 1
        elif class_decided == no_diagnosis_color:
            N_not_found +=1
            not_found.append(subject_diagnoses)

    print("Statistics")
    print("No list: ", N_no_list)
    print("Empty list: ", N_empty_list)
    print("Not found: ", N_not_found)
    print(not_found)

    return colors


def get_diagnoses(subject_codes) -> dict[str, list[str]]:

    # Read diagnoses
    diagnoses = read_diagnoses('KJPP')

    # Statistics counters
    N_no_list, N_empty_list, N_not_found, N_exists = 0, 0, 0, 0

    res = {}
    for code in subject_codes:
        code = code.split('$')[0]
        if code not in diagnoses:
            N_not_found += 1
            continue
        subject_diagnoses = diagnoses[code]
        res[code] = []
        if isinstance(subject_diagnoses, list):
            if len(subject_diagnoses) == 0:
                N_empty_list += 1
            else:
                N_exists += 1
                for d in subject_diagnoses:
                    if icd.is_valid_item(d):
                        res[code].append(d)
        else:
            N_no_list += 1

    print("Statistics")
    print("Not found (no age or gender):", N_not_found)
    print("No list (no report):", N_no_list)
    print("Empty list (no diagnoses):", N_empty_list)
    print("Exists:", N_exists)
    print("Total:", N_not_found + N_no_list + N_empty_list + N_exists)
    print("Total subjects:", len(subject_codes))
    print()

    return res


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

