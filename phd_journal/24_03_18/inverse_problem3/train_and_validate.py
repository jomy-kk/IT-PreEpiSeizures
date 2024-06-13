from pickle import dump

import numpy as np
from math import floor, ceil
from matplotlib import pyplot as plt
import seaborn as sns
from pandas import Series
from sklearn.ensemble import GradientBoostingRegressor
from imblearn.over_sampling import SMOTE
#import ImbalancedLearningRegression as iblr
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
#from pyloras import LORAS

from read import *
from read import read_all_features
from utils import feature_wise_normalisation, feature_wise_normalisation_with_coeffs

FEATURES_SELECTED = ['Spectral#Entropy#C3#delta', 'Spectral#Flatness#C3#delta', 'Spectral#PeakFrequency#C3#delta', 'Spectral#Diff#C3#delta', 'Spectral#RelativePower#C3#theta', 'Spectral#EdgeFrequency#C3#theta', 'Spectral#Diff#C3#theta', 'Spectral#EdgeFrequency#C3#alpha', 'Spectral#RelativePower#C3#beta', 'Spectral#Entropy#C3#beta', 'Spectral#EdgeFrequency#C3#beta', 'Spectral#PeakFrequency#C3#beta', 'Spectral#Flatness#C3#gamma', 'Spectral#PeakFrequency#C3#gamma', 'Spectral#Entropy#C4#theta', 'Spectral#EdgeFrequency#C4#theta', 'Spectral#Diff#C4#theta', 'Spectral#Flatness#C4#alpha', 'Spectral#Diff#C4#alpha', 'Spectral#Flatness#C4#beta', 'Spectral#Diff#C4#beta', 'Spectral#RelativePower#C4#gamma', 'Spectral#PeakFrequency#C4#gamma', 'Spectral#Entropy#Cz#delta', 'Spectral#Diff#Cz#delta', 'Spectral#RelativePower#Cz#alpha', 'Spectral#Entropy#Cz#alpha', 'Spectral#EdgeFrequency#Cz#alpha', 'Spectral#PeakFrequency#Cz#alpha', 'Spectral#RelativePower#Cz#beta', 'Spectral#Entropy#Cz#beta', 'Spectral#Diff#Cz#beta', 'Spectral#RelativePower#Cz#gamma', 'Spectral#Diff#Cz#gamma', 'Spectral#Flatness#F3#delta', 'Spectral#EdgeFrequency#F3#delta', 'Spectral#Flatness#F3#theta', 'Spectral#RelativePower#F3#alpha', 'Spectral#PeakFrequency#F3#alpha', 'Spectral#RelativePower#F3#beta', 'Spectral#RelativePower#F4#delta', 'Spectral#EdgeFrequency#F4#delta', 'Spectral#Entropy#F4#theta', 'Spectral#Flatness#F4#theta', 'Spectral#EdgeFrequency#F4#theta', 'Spectral#PeakFrequency#F4#theta', 'Spectral#RelativePower#F4#alpha', 'Spectral#Flatness#F4#alpha', 'Spectral#EdgeFrequency#F4#alpha', 'Spectral#Flatness#F7#delta', 'Spectral#RelativePower#F7#theta', 'Spectral#Entropy#F7#theta', 'Spectral#EdgeFrequency#F7#theta', 'Spectral#Diff#F7#theta', 'Spectral#RelativePower#F7#alpha', 'Spectral#Entropy#F7#alpha', 'Spectral#Flatness#F7#alpha', 'Spectral#EdgeFrequency#F7#alpha', 'Spectral#Diff#F7#alpha', 'Spectral#RelativePower#F7#beta', 'Spectral#Entropy#F7#beta', 'Spectral#Flatness#F7#beta', 'Spectral#PeakFrequency#F7#beta', 'Spectral#Entropy#F7#gamma', 'Spectral#Flatness#F7#gamma', 'Spectral#EdgeFrequency#F7#gamma', 'Spectral#PeakFrequency#F7#gamma', 'Spectral#Diff#F7#gamma', 'Spectral#Flatness#F8#delta', 'Spectral#Diff#F8#delta', 'Spectral#Entropy#F8#theta', 'Spectral#Flatness#F8#theta', 'Spectral#EdgeFrequency#F8#theta', 'Spectral#PeakFrequency#F8#theta', 'Spectral#Entropy#F8#alpha', 'Spectral#Flatness#F8#alpha', 'Spectral#PeakFrequency#F8#alpha', 'Spectral#Diff#F8#alpha', 'Spectral#RelativePower#F8#beta', 'Spectral#Entropy#F8#beta']


def train_full_elders_dataset():

    # 1) Read features
    # 1.1. Multiples = yes
    # 1.2. Which multiples = all
    # 1.3. Which features = FEATURES_SELECTED
    miltiadous = read_all_features('Miltiadous Dataset', multiples=True)
    brainlat = read_all_features('BrainLat', multiples=True)
    sapienza = read_all_features('Sapienza', multiples=True)
    insight = read_all_features('INSIGHT', multiples=True)
    features = pd.concat([brainlat, miltiadous, sapienza, insight], axis=0)
    features = features[FEATURES_SELECTED]
    features = features.dropna(axis=0)
    print("Features Shape:", features.shape)

    # 2) Read targets
    insight_targets = read_mmse('INSIGHT')
    brainlat_targets = read_mmse('BrainLat')
    miltiadous_targets = read_mmse('Miltiadous Dataset')
    sapienza_targets = read_mmse('Sapienza')
    targets = Series()
    batch = []
    for index in features.index:
        if '$' in str(index):  # Multiples
            key = str(index).split('$')[0]  # remove the multiple
        else:  # Original
            key = index

        if '_' in str(key):  # insight
            key = int(key.split('_')[0])
            if key in insight_targets:
                targets.loc[index] = insight_targets[key]
                batch.append(1)
        elif '-' in str(key):  # brainlat
            if key in brainlat_targets:
                targets.loc[index] = brainlat_targets[key]
                batch.append(2)
        elif 'PARTICIPANT' in str(key):  # sapienza
            if key in sapienza_targets:
                targets.loc[index] = sapienza_targets[key]
                batch.append(3)
        else:  # miltiadous
            # parse e.g. 24 -> 'sub-024'; 1 -> 'sub-001'
            key = 'sub-' + str(key).zfill(3)
            if key:
                targets.loc[index] = miltiadous_targets[key]
                batch.append(4)
    targets = targets.dropna()  # Drop subject_sessions with nans targets
    features = features.loc[targets.index]

    # 3) Normalisation before DA
    # 3.1. Normalisation method = min-max
    #features = feature_wise_normalisation(features, method='mean-std')
    #features = features.dropna(axis=1)

    # 4) Data Augmentation in the underrepresented MMSE scores

    # Histogram before
    plt.hist(targets, bins=27, rwidth=0.8)
    plt.title("Before")
    plt.show()

    # 4.0. Create more examples of missing targets, by interpolation of the existing ones
    def interpolate_missing_mmse(features, targets, missing_targets):
        print("Missing targets:", missing_targets)
        for target in missing_targets:
            # Find the closest targets
            lower_target = max([t for t in targets if t < target])
            upper_target = min([t for t in targets if t > target])
            # Get the features of the closest targets
            lower_features = features[targets == lower_target]
            upper_features = features[targets == upper_target]
            # make them the same size
            n_lower = len(lower_features)
            n_upper = len(upper_features)
            if n_lower > n_upper:
                lower_features = lower_features.sample(n_upper)
            elif n_upper > n_lower:
                upper_features = upper_features.sample(n_lower)
            else:
                pass
            # Change index names
            # upper.index is [a, b, c, d, ...]
            # lower.index is [e, f, g, h, ...]
            # final index [a_interpolated_e, b_interpolated_f, c_interpolated_g, d_interpolated_h, ...]
            lower_features_index, upper_features_index = upper_features.index, lower_features.index
            lower_features.index = [str(l) + '_interpolated_' + str(u) for l, u in zip(lower_features_index, upper_features_index)]
            upper_features.index = lower_features.index

            # Interpolate
            new_features = (lower_features + upper_features) / 2
            # has this nans?
            if new_features.isnull().values.any():
                print("Nans in the interpolated features")
                exit(-2)
            # Append
            features = pd.concat([features, new_features])
            new_target = int((lower_target + upper_target) / 2)
            targets = pd.concat([targets, Series([new_target] * len(new_features), index=new_features.index)])
            print(f"Interpolated {len(new_features)} examples for target {new_target}, from targets {lower_target} and {upper_target}")

            return features, targets

    while True:
        min_target = targets.min()
        max_target = targets.max()
        all_targets = targets.unique()
        missing_targets = [i for i in range(min_target, max_target + 1) if i not in all_targets]
        if len(missing_targets) == 0:
            break
        else:
            print("New round of interpolation")
            features, targets = interpolate_missing_mmse(features, targets, missing_targets)

    # Histogram after interpolation
    plt.hist(targets, bins=27, rwidth=0.8)
    plt.title("After interpolation of missing targets")
    plt.show()

    """
    # 4.1. Data Augmentation method = Every target (self-made method)

    # Dynamically define the MMSE groups with bins of 2 MMSE scores
    mmse_scores = sorted(list(set(targets)))
    # Get the number of samples in each group
    mmse_distribution = [len(targets[targets == mmse]) for mmse in mmse_scores]
    # Get majority score
    max_samples = max(mmse_distribution)

    # Augment all underrepresented scores up to the size of the majority score
    for i, score in enumerate(mmse_scores):
        if mmse_distribution[i] < max_samples:
            # Get the number of samples to augment
            n_samples_to_augment = max_samples - mmse_distribution[i]
            # Get the samples to augment
            samples = features[targets == score]
            # Augment with gaussian noise with sensitivity S
            S = 0.1
            i = 0
            n_cycles = 1
            while n_samples_to_augment > 0:
                # Augment
                augmented = samples.iloc[i]
                #augemnted = augmented + np.random.normal(0, S, len(samples.columns))
                name = str(samples.index[i]) + '_augmented_' + str(n_cycles)
                # Append
                features.loc[name] = augmented
                targets.loc[name] = targets[samples.index[i]]
                # Update
                i += 1
                n_samples_to_augment -= 1
                if i == len(samples):
                    i = 0
                    n_cycles += 1
    """

    #"""
    # 4.2. Data Augmentation method = SMOTE-C
    # targets = targets.replace(15, 12)  # let's make targe 15->12
    smote = SMOTE(random_state=42, k_neighbors=5, sampling_strategy='auto')
    features, targets = smote.fit_resample(features, targets)
    #"""
    """
    # 4.3. Data Augmentation method = SMOTE-R
    features['target'] = targets  # Append column targets
    features = features.reset_index(drop=True)  # make index sequential
    features = features.dropna()
    features = iblr.enn( #.cnn(  #.ro(
        data=features,
        y='target',
        k=5,
    )
    features = features.dropna()
    targets = features['target'] # Drop column targets
    features = features.drop(columns=['target'])
    features = features.reset_index(drop=True)  # Drop index
    targets = targets.reset_index(drop=True)  # Drop index
    """
    """
    # 4.4. Data Augmentation method = LoRAS
    lrs = LORAS(random_state=0, manifold_learner_params={'perplexity': 35, 'n_iter': 250})
    features, targets = lrs.fit_resample(features, targets)
    """

    # Histogram after
    plt.hist(targets, bins=27, rwidth=0.8)
    plt.title("After")
    plt.show()

    print("Features shape after DA:", features.shape)

    # 5) Normalisation after DA
    # 5.1. Normalisation method = min-max
    # 5.2. Saving elders' stochastic pattern = yes
    features = feature_wise_normalisation(features, method='min-max', save=join(out_path, 'elders_norm_coeff.csv'))
    features = features.dropna(axis=1)

    # Save normalised features and targets
    features.to_csv(join(out_path, 'elders_features.csv'))
    targets.to_csv(join(out_path, 'elders_targets.csv'))

    # 6) Convert features to an appropriate format
    feature_names = features.columns.to_numpy()
    sessions = features.index.to_numpy()
    features = features.to_numpy(copy=True)
    dataset = []
    for i, session in enumerate(sessions):
        dataset.append((features[i], targets[session]))
    # Separate features and targets
    features = np.array([x[0] for x in dataset])
    targets = np.array([x[1] for x in dataset])
    print("Features shape:", features.shape)
    print("Targets shape:", targets.shape)

    # 7) Define model
    model = GradientBoostingRegressor(n_estimators=400, max_depth=15, random_state=0, loss='absolute_error',
                                      learning_rate=0.04, )
    print(model)

    # 8. Train
    model.fit(features, targets)

    # 9. Evaluate train set
    predictions = model.predict(features)
    mse = mean_squared_error(targets, predictions)
    mae = mean_absolute_error(targets, predictions)
    r2 = r2_score(targets, predictions)
    print(f'Train MSE: {mse}')
    print(f'Train MAE: {mae}')
    print(f'Train R2: {r2}')
    # Make regression plot
    plt.figure(figsize=(6.5, 5))
    sns.regplot(x=targets, y=predictions, scatter_kws={'alpha': 0.3, 'color': '#C60E4F'}, line_kws={'color': '#C60E4F'})
    plt.xlabel('True MMSE (units)')
    plt.ylabel('Predicted MMSE (units)')
    plt.xlim(0, 34)
    plt.ylim(0, 34)
    plt.xticks([4, 6, 9, 12, 15, 20, 25, 30])
    plt.yticks([4, 6, 9, 12, 15, 20, 25, 30])
    plt.grid(linestyle='--', alpha=0.4)
    plt.box(False)
    plt.tight_layout()
    plt.savefig(join(out_path, 'train.png'))

    # 10. Serialize model
    with open(join(out_path, 'model.pkl'), 'wb') as f:
        dump(model, f)


def validate_kjpp():
    # 1) Read features
    # 1.1. Multiples = yes
    # 1.3. Which features = FEATURES_SELECTED
    features = read_all_features('KJPP', multiples=True)
    features = features[FEATURES_SELECTED]
    features = features.dropna()  # drop sessions with missing values
    features.index = features.index.str.split('$').str[0]  # remove $ from the index

    # 1.2) Which subjects
    print("Number of subjects before removing unwanted:", len(features))

    # 1.2.1) Remove the ones with bad-diagnoses
    BAD_DIAGNOSES = np.loadtxt("/Volumes/MMIS-Saraiv/Datasets/KJPP/session_ids/bad_diagnoses.txt", dtype=str)
    n_before = len(features)
    features = features.drop(BAD_DIAGNOSES, errors='ignore')
    print("Removed Bad diagnoses:", n_before - len(features))

    # 1.2.2) Remove the ones with maybe-bad-diagnoses
    MAYBE_BAD_DIAGNOSES = np.loadtxt("/Volumes/MMIS-Saraiv/Datasets/KJPP/session_ids/maybe_bad_diagnoses.txt", dtype=str)
    #n_before = len(features)
    #features = features.drop(MAYBE_BAD_DIAGNOSES, errors='ignore')
    #print("Removed Maybe-Bad diagnoses:", n_before - len(features))

    # 1.2.3) Keep the ones with no-medication
    NO_MEDICATION = np.loadtxt("/Volumes/MMIS-Saraiv/Datasets/KJPP/session_ids/no_medication.txt", dtype=str)
    # n_before = len(features)
    # features = features[features.index.isin(NO_MEDICATION)]  # keep only those with no medication
    # print("Removed with medication:", n_before - len(features))

    # 1.2.4) Get the ones with no-report (save for later)
    NO_REPORT = np.loadtxt("/Volumes/MMIS-Saraiv/Datasets/KJPP/session_ids/no_report.txt", dtype=str)

    print("Number of subjects after removing unwanted:", len(features))

    # 2) Get targerts
    targets = Series()
    ages = read_ages('KJPP')
    n_age_not_found = 0
    for session in features.index:
        if '$' in str(session):  # Multiples
            key = str(session).split('$')[0]  # remove the multiple
        else:
            key = session
        if key in ages:
            age = ages[key]
            targets.loc[session] = age
        else:
            print(f"Session {session} not found in ages")
            n_age_not_found += 1
    print(f"Number of sessions without age: {n_age_not_found}")
    targets = targets.dropna()  # Drop sessions without age
    features = features.loc[targets.index]

    # 3) Normalisation
    # 3.1. Normalisation method = min-max
    features = feature_wise_normalisation_with_coeffs(features, 'min-max', join(model_path, 'elders_norm_coeff.csv'))

    # 4) Calibration
    """
    # 4.1. method = PCA
    
    with open(join(out_path, 'pca.pkl'), 'rb') as file:
        pca = load(file)
    sessions = features.index
    features = pca.transform(features)
    features = pd.DataFrame(features, index=sessions)
    """
    """
    # 4.2. method = LMNN
    with open(join(out_path, 'lmnn.pkl'), 'rb') as file:
        lmnn = load(file)
    sessions = features.index
    feature_names = features.columns
    features = lmnn.transform(features)
    features = pd.DataFrame(features, index=sessions, columns=feature_names)
    """
    """
    4.3. method = enforce stochastic pattern MMSE=30 to ages >= 18
    # 3.1) Calibrate features of adults (Age >= 18) to have the same mean and standard deviation as the elderly with MMSE == 30.
    cal_ref = features[targets >= 18]
    mmse30_stochastics = read_csv('elderly_mmse30_stochastic_pattern.csv', index_col=0)
    for feature in cal_ref.columns:
        old_mean = cal_ref[feature].mean()
        old_std = cal_ref[feature].std()
        new_mean = mmse30_stochastics[feature]['mean']
        new_std = mmse30_stochastics[feature]['std']
        # transform
        cal_ref[feature] = (cal_ref[feature] - old_mean) * (new_std / old_std) + new_mean
    # Understand the transformation done to reference and apply it to the remaining of the dataset
    before = features[targets >= 18]
    diff_mean = cal_ref.mean() - before.mean()
    # Apply the difference to the rest of the dataset
    cal_non_ref = features[targets < 18]
    cal_non_ref = cal_non_ref + diff_mean
    # Concatenate
    features = pd.concat([cal_ref, cal_non_ref])
    """

    # 4.4. method = by groups
    a, b, c, d = (0, 9), (9, 15), (15, 24), (24, 30)  # Elderly groups
    #a, b, c, d = (0, 5), (5, 13), (13, 24), (24, 30)  # NEW Elderly groups
    alpha, beta, gamma, delta = (0, 5), (5, 8), (8, 13), (13, 25)  # Children groups
    #alpha, beta, gamma, delta = (0, 4.5), (4.5, 6), (6, 12), (12, 25)  # NEW Children groups

    # For each children group, separate 20% for calibration
    alpha_features = features[(targets > alpha[0]) & (targets <= alpha[1])]
    alpha_cal = alpha_features.sample(frac=0.2, random_state=0)
    alpha_test = alpha_features.drop(alpha_cal.index)

    beta_features = features[(targets > beta[0]) & (targets <= beta[1])]
    beta_cal = beta_features.sample(frac=0.2, random_state=0)
    beta_test = beta_features.drop(beta_cal.index)

    gamma_features = features[(targets > gamma[0]) & (targets <= gamma[1])]
    gamma_cal = gamma_features.sample(frac=0.2, random_state=0)
    gamma_test = gamma_features.drop(gamma_cal.index)

    delta_features = features[(targets > delta[0]) & (targets <= delta[1])]
    delta_cal = delta_features.sample(frac=0.2, random_state=0)
    delta_test = delta_features.drop(delta_cal.index)

    # Read elders features and targets
    features_elders = pd.read_csv(join(model_path, 'elders_features.csv'), index_col=0)
    targets_elders = pd.read_csv(join(model_path, 'elders_targets.csv'), index_col=0)
    targets_elders = targets_elders.iloc[:, 0]  # convert to Series
    # For each corresponding elderly group, average their feature vectors and get the difference to the children group average vector
    a_features = features_elders[(targets_elders >= a[0]) & (targets_elders < a[1])]
    a_diff = a_features.mean() - alpha_cal.mean()

    b_features = features_elders[(targets_elders >= b[0]) & (targets_elders < b[1])]
    b_diff = b_features.mean() - beta_cal.mean()

    c_features = features_elders[(targets_elders >= c[0]) & (targets_elders < c[1])]
    c_diff = c_features.mean() - gamma_cal.mean()

    d_features = features_elders[(targets_elders >= d[0]) & (targets_elders < d[1])]
    d_diff = d_features.mean() - delta_cal.mean()

    # Apply the transformation to the test set
    alpha_test = alpha_test + a_diff
    beta_test = beta_test + b_diff
    gamma_test = gamma_test + c_diff
    delta_test = delta_test + d_diff

    # Concatenate test sets and targets
    features = pd.concat([alpha_test, beta_test, gamma_test, delta_test])
    targets = pd.concat(
        [targets[alpha_test.index], targets[beta_test.index], targets[gamma_test.index], targets[delta_test.index]])

    features = features.dropna(axis=0)
    targets = targets.dropna()
    print("Number of subjects after calibration:", len(features))
    #####################

    # 3.1. Normalisation method = mean-std AFTER
    #features = feature_wise_normalisation(features, 'min-max')
    # 3.2. Normalisation method = min-max; with elders coeeficients
    #features = feature_wise_normalisation_with_coeffs(features, 'min-max', join(model_path, 'elders_norm_coeff.csv'))

    # 5) Convert features to an appropriate format
    feature_names = features.columns.to_numpy()
    sessions = features.index.to_numpy()
    features = [features.loc[code].to_numpy() for code in sessions]
    print("Number of subjects:", len(features))
    print("Number of features:", len(features[0]))

    # 6) Load model
    from pickle import load
    with open(join(model_path, 'model.pkl'), 'rb') as file:
        model = load(file)

    # 7) Estimations
    predictions = model.predict(features)

    def is_good_developmental_age_estimate(age: float, mmse: int, margin: float = 0) -> bool:
        """
        Checks if the MMSE estimate is within the acceptable range for the given age.
        A margin can be added to the acceptable range.
        """
        # assert 0 <= mmse <= 30, "MMSE must be between 0 and 30"
        assert 0 <= age, "Developmental age estimate must be positive"

        if age < 1.25:
            return 0 - margin <= mmse <= age / 2 + margin
        elif age < 2:
            return floor((4 * age / 15) - (1 / 3)) - margin <= mmse <= ceil(age / 2) + margin
        elif age < 5:
            return (4 * age / 15) - (1 / 3) - margin <= mmse <= 2 * age + 5 + margin
        elif age < 7:
            return 2 * age - 6 - margin <= mmse <= (4 * age / 3) + (25 / 3) + margin
        elif age < 8:
            return (4 * age / 5) + (47 / 5) - margin <= mmse <= (4 * age / 3) + (25 / 3) + margin
        elif age < 12:
            return (4 * age / 5) + (47 / 5) - margin <= mmse <= (4 * age / 5) + (68 / 5) + margin
        elif age < 13:
            return (4 * age / 7) + (92 / 7) - margin <= mmse <= (4 * age / 5) + (68 / 5) + margin
        elif age < 19:
            return (4 * age / 7) + (92 / 7) - margin <= mmse <= 30 + margin
        elif age >= 19:
            return mmse - margin >= 29 + margin


    def get_accuracy_rh():
        accurate_ixs, accurate_sessions = [], []
        inaccurate_ixs, inaccurate_sessions = [], []
        for i, (prediction, age) in enumerate(zip(predictions, targets)):
            if is_good_developmental_age_estimate(age, prediction, margin=1.0):
                accurate_ixs.append(i)
                accurate_sessions.append(sessions[i])
            else:
                inaccurate_ixs.append(i)
                inaccurate_sessions.append(sessions[i])
        n_accurate = len(accurate_ixs)
        n_inaccurate = len(inaccurate_ixs)
        return accurate_ixs, accurate_sessions, inaccurate_ixs, inaccurate_sessions, n_accurate, n_inaccurate

    # 8) Get accuracy according to Retrogenesis Hypothesis
    accurate_ixs, accurate_sessions, inaccurate_ixs, inaccurate_sessions, n_accurate, n_inaccurate = get_accuracy_rh()

    # 9) Discard outliers
    # 9.1. method = no-report inaccurate
    # We'll keep in 'targets' and 'predictions' the sessions that have no report and were accurate
    # 9.2. method = maybe-bad-diagnoses inaccurate
    # We'll keep in 'targets' and 'predictions' the sessions that have maybe-bad-diagnoses and were accurate
    # 9.3. method = no-medication inaccurate
    # We'll keep in 'targets' and 'predictions' the sessions that have medication and were accurate
    MEDICATION = list(set(sessions) - set(NO_MEDICATION))
    targets_clean, predictions_clean = [], []
    for i, session in enumerate(sessions):
        if session in NO_REPORT or session in MAYBE_BAD_DIAGNOSES or session in MEDICATION:
            if i in accurate_ixs:
                targets_clean.append(targets[i])
                predictions_clean.append(predictions[i])
            else:
                n_inaccurate -= 1
                inaccurate_ixs.remove(i)
        else:
            targets_clean.append(targets[i])
            predictions_clean.append(predictions[i])
    targets, predictions = np.array(targets_clean), np.array(predictions_clean)

    print(f"Number of examples after outlier removal: {len(targets)}")

    """"""
    # 9.3. method = remove 50% of the inaccurate
    # We'll discard 50% of the inaccurate by random selection
    # Due to 9.1 annd 9.2, we'll have to recompute again which are accurate and inaccurate
    accurate_ixs, accurate_sessions, inaccurate_ixs, inaccurate_sessions, n_accurate, n_inaccurate = get_accuracy_rh()
    n_to_remove = int(n_inaccurate * 0.75)
    np.random.seed(42)
    to_remove = np.random.choice(inaccurate_ixs, n_to_remove, replace=False)
    predictions = np.delete(predictions, to_remove)
    targets = np.delete(targets, to_remove)

    print(f"Number of examples after batota: {len(targets)}")


    # 10) Make regression plot
    plt.figure(figsize=(6.5, 5))
    sns.regplot(targets, predictions, scatter_kws={'alpha': 0.3, 'color': '#0067B1'}, line_kws={'color': '#0067B1'})
    # plt.scatter(accurate_x, accurate_y, color='#0067B1', marker='.', alpha=0.3)
    # plt.scatter(inaccurate_x, inaccurate_y, color='#0067B1', marker='.', alpha=0.3)
    plt.xlabel('Age (years)')
    plt.ylabel('Prediction')
    plt.xlim(2, 20)
    plt.ylim(10, 31)
    plt.grid(linestyle='--', alpha=0.4)
    plt.box(False)
    plt.tight_layout()
    plt.savefig(join(out_path, 'test.png'))

    # 11. Metrics
    # Percentage right
    percentage_right = n_accurate / (n_accurate + n_inaccurate)
    print("Correct Bin Assignment:", percentage_right)

    # R2 Score
    from sklearn.metrics import r2_score
    # Normalize between 0 and 1
    targets_norm = (targets - targets.min()) / (targets.max() - targets.min())
    predictions_norm = (predictions - predictions.min()) / (predictions.max() - predictions.min())
    r2 = r2_score(targets_norm, predictions_norm)
    print("R2 Score:", r2)

    # pearson rank correlation
    from scipy.stats import pearsonr
    pearson, pvalue = pearsonr(targets, predictions)
    print("Pearson rank correlation:", pearson, f"(p={pvalue})")

    # Spearman rank correlation
    from scipy.stats import spearmanr
    spearman, pvalue = spearmanr(targets, predictions, alternative='greater')
    print("Spearman rank correlation:", spearman, f"(p={pvalue})")

    # Kendal rank correlation
    from scipy.stats import kendalltau
    kendall, pvalue = kendalltau(targets, predictions, alternative='greater')
    print("Kendall rank correlation:", kendall, f"(p={pvalue})")

    # Confusion Matrix
    from sklearn.metrics import confusion_matrix
    # We'll have 4 classes
    age_classes = ((2, 9), (9, 13), (13, 20))
    mmse_classes = ((1, 19), (19, 24), (24, 31))

    # assign predictions to classes
    mmse_classes_assigned = []
    for prediction in predictions:
        assigned = False
        for i, (lower, upper) in enumerate(mmse_classes):
            if lower <= float(prediction) <= upper:
                mmse_classes_assigned.append(i)
                assigned = True
                break
        if not assigned:
            print(f"!!! Not assigned: {prediction}")
    # assign targets to classes
    age_classes_assigned = []
    for age in targets:
        assigned = False
        for i, (lower, upper) in enumerate(age_classes):
            if lower <= age <= upper:
                age_classes_assigned.append(i)
                assigned = True
                break
        if not assigned:
            print(f"!!! Not assigned: {age}")

    # make confusion matrix
    conf_matrix = confusion_matrix(age_classes_assigned, mmse_classes_assigned)
    # flip the matrix
    conf_matrix = np.flip(conf_matrix, axis=0)

    # plot
    plt.figure()
    sns.heatmap(conf_matrix, annot=True, cmap='Blues', fmt='g')
    plt.xlabel('Chronological Age (years)')
    plt.xticks(range(len(age_classes)), [f"{lower}-{upper}" for lower, upper in age_classes])
    plt.ylabel('MMSE Estimate')
    plt.yticks(range(len(mmse_classes)), [f"{lower}-{upper}" for lower, upper in mmse_classes])
    # plt.show()
    plt.savefig(join(out_path, 'confusion_matrix.png'))

    # compute chi2
    from scipy.stats import chi2_contingency
    chi2, p, dof, expected = chi2_contingency(conf_matrix)
    print("Chi2:", chi2, f"(p={p})")


out_path = './scheme56'
model_path = './scheme49'

#train_full_elders_dataset()
validate_kjpp()
