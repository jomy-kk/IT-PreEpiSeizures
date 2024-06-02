# Test if there is data leakage in the multiple examples generated from the same subject-session.
from glob import glob
from os.path import join

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.spatial.distance import euclidean
from scipy.stats import ttest_ind

# Get DataFrames
common_path = '/Volumes/MMIS-Saraiv/Datasets/Miltiadous Dataset/features'

FEATURES_SELECTED = ['Spectral#Diff#C3#theta', 'Spectral#RelativePower#C3#gamma', 'Spectral#RelativePower#C4#delta', 'Spectral#PeakFrequency#C4#beta1', 'Spectral#Flatness#C4#gamma', 'Spectral#RelativePower#Cz#beta2', 'Spectral#RelativePower#Cz#beta3', 'Spectral#RelativePower#Cz#gamma', 'Spectral#Diff#Cz#gamma', 'Spectral#Entropy#F4#beta2', 'Spectral#Flatness#F4#beta2', 'Spectral#Entropy#F7#theta', 'Spectral#PeakFrequency#Fp2#alpha1', 'Spectral#PeakFrequency#Fp2#beta2', 'Spectral#RelativePower#Fz#delta', 'Spectral#PeakFrequency#Fz#theta', 'Spectral#PeakFrequency#Fz#gamma', 'Spectral#PeakFrequency#O1#beta3', 'Spectral#Entropy#O2#delta', 'Spectral#PeakFrequency#O2#theta', 'Spectral#PeakFrequency#P3#gamma', 'Spectral#Diff#P4#beta2', 'Spectral#EdgeFrequency#Pz#gamma', 'Spectral#EdgeFrequency#T3#delta', 'Spectral#Flatness#T5#alpha2', 'Hjorth#Activity#C3', 'Hjorth#Activity#P4', 'Hjorth#Mobility#Cz', 'PLI#Temporal(L)-Parietal(L)#alpha2', 'PLI#Temporal(L)-Occipital(L)#beta1']

# Iterate directories
all_directories = glob(join(common_path, '*'))
all_directories = [directory for directory in all_directories if not directory.endswith('.csv')]

sessions_with_multiple_examples = []
sessions_with_single_examples = []
for subject_path in all_directories:
    all_multiple_files = glob(join(subject_path, '*$Multiple.csv'))
    if len(all_multiple_files) != 0:  # underrepresented target
        # Read them all
        all_data = [pd.read_csv(file, index_col=0) for file in all_multiple_files]
        # Concatenate all their columns in a single DataFrame
        all_data = pd.concat(all_data, axis=1)
        # Keep only selected features
        FEATURES_SELECTED = [col for col in all_data.columns if col in FEATURES_SELECTED]
        selected_features = all_data[FEATURES_SELECTED]
        # drop nans
        selected_features = selected_features.dropna()
        sessions_with_multiple_examples.append(selected_features)

        # DISTORTION / AUGMENTATION
        """
        def jitter(data, sigma=0.05):
            myNoise = np.random.normal(loc=0, scale=sigma, size=data.shape)
            return data + myNoise

        def scaling(data, sigma=0.1):
            scalingFactor = np.random.normal(loc=1.0, scale=sigma,
                                             size=(data.shape[0]))  # shape=(1,-1) for multivariate data
            return data * scalingFactor

        # Apply jittering and scaling to each feature
        for feature in selected_features.columns:
            data = selected_features[feature].values
            data = jitter(data, sigma=0.1)
            data = scaling(data, sigma=0.04)
            selected_features[feature] = data
        """

        # VARIANCE AND DISTANCE
        """
        # 1. Compute variance by feature
        variances = selected_features.var()
        print("Variance by feature:")
        print(variances)
        print("Average variance:", np.mean(variances))
        means = selected_features.mean()
        print("Mean by feature:", means)
        print("Average mean:", np.mean(means))

        # 2. Compute Euclidean distance between every feature vector pair of examples, and then average them.
        euclidean_distances = []
        for i in range(len(selected_features)):
            for j in range(i + 1, len(selected_features)):
                euclidean_distances.append(euclidean(selected_features.iloc[i], selected_features.iloc[j]))
        average_euclidean_distance = np.mean(euclidean_distances)
        print("Average Euclidean distance:", average_euclidean_distance)
        """

        # STATISTICAL TESTS
        """
        # 3. Perform t-test between every feature vector pair of examples, and then average them.
        print("T-test")
        t_statistics = []
        p_values = []
        for i in range(len(selected_features)):
            for j in range(i + 1, len(selected_features)):
                t_stat, p_val = ttest_ind(selected_features.iloc[i], selected_features.iloc[j])
                t_statistics.append(t_stat)
                p_values.append(p_val)
        average_t_statistic = np.mean(t_statistics)
        average_p_value = np.mean(p_values)
        print("Average t-statistic:", average_t_statistic)
        print("Average p-value:", average_p_value)

        # 4. ANOVA
        from scipy.stats import f_oneway
        print("ANOVA test")
        for feature in selected_features.columns:
            # Split data into separate arrays for each example
            data = [selected_features[feature].values for _ in range(len(selected_features))]
            # Perform ANOVA test
            stat, p = f_oneway(*data)
            print(f'Feature: {feature}, Statistics={stat}, p={p}')

            # interpret
            alpha = 0.05
            if p > alpha:
                print(f'{feature}: Same distribution (fail to reject H0)')
            else:
                print(f'{feature}: Different distribution (reject H0)')


        # 5. Mann-Whitney U test (non-parametric)
        from scipy.stats import mannwhitneyu
        print("Mann-Whitney U test")
        for feature in selected_features.columns:
            # Split data into two samples
            data1 = selected_features[feature][:len(selected_features) // 2]
            data2 = selected_features[feature][len(selected_features) // 2:]
            # Perform Mann-Whitney U test
            stat, p = mannwhitneyu(data1, data2)
            print(f'Feature: {feature}, Statistics={stat}, p={p}')

            # interpret
            alpha = 0.05
            if p > alpha:
                print(f'{feature}: Same distribution (fail to reject H0)')
            else:
                print(f'{feature}: Different distribution (reject H0)')

        # 6. Kruskal-Wallis test (non-parametric)
        from scipy.stats import kruskal
        print("Kruskal-Wallis test")
        for feature in selected_features.columns:
            # Split data into separate arrays for each example
            data = [selected_features[feature].values for _ in range(len(selected_features))]
            # Perform Kruskal-Wallis H-test
            stat, p = kruskal(*data)
            print(f'Feature: {feature}, Statistics={stat}, p={p}')

            # interpret
            alpha = 0.05
            if p > alpha:
                print(f'{feature}: Same distribution (fail to reject H0)')
            else:
                print(f'{feature}: Different distribution (reject H0)')
        """

    else:  # overrepresented target
        all_single_files = glob(join(subject_path, 'Hjorth*.csv')) + glob(join(subject_path, 'Spectral*.csv')) + glob(join(subject_path, 'PLI*.csv')) + glob(join(subject_path, 'Connectivity*.csv'))
        all_data = []
        for file in all_single_files:
            if "Spectral" in file:
                df = pd.read_csv(file)
            else:
                df = pd.read_csv(file, index_col=0)
            df.index = [subject_path.split('/')[-1], ]
            all_data.append(df)
        all_data = pd.concat(all_data, axis=1)
        FEATURES_SELECTED = [col for col in all_data.columns if col in FEATURES_SELECTED]
        selected_features = all_data[FEATURES_SELECTED]
        selected_features = selected_features.dropna()
        sessions_with_single_examples.append(selected_features)

print("Sessions with multiple examples:", len(sessions_with_multiple_examples))
print("Sessions with single examples:", len(sessions_with_single_examples))

# Make DataFrames
all_multiple_examples = pd.concat(sessions_with_multiple_examples)
all_single_examples = pd.concat(sessions_with_single_examples)

# Normalize min-max by feature, using joint minimum and maximum
min_max = pd.concat([all_multiple_examples, all_single_examples]).agg(['min', 'max'])
all_multiple_examples = (all_multiple_examples - min_max.loc['min']) / (min_max.loc['max'] - min_max.loc['min'])
all_single_examples = (all_single_examples - min_max.loc['min']) / (min_max.loc['max'] - min_max.loc['min'])

# Compare distances
from scipy.spatial.distance import euclidean
import matplotlib.pyplot as plt

# Initialize a list to store the average distances
average_distances = []

# Iterate over the 8 sessions with multiple examples
for session in sessions_with_multiple_examples[:8]:
    # Iterate over the examples in the current session
    for i in range(len(session)):
        # Initialize a list to store the distances for the current example
        distances = []
        # Iterate over the other 80 subjects
        for other_session in sessions_with_single_examples:
            # Compute the Euclidean distance to each example from the other subject
            for j in range(len(other_session)):
                distance = euclidean(session.iloc[i], other_session.iloc[j])
                distances.append(distance)
        # Compute the average distance for the current example
        average_distance = np.mean(distances)
        # Add the average distance to the list
        average_distances.append(average_distance)

# Plot the average distances
plt.figure(figsize=(10, 6))
plt.plot(average_distances, label='Other-Subject')

# Initialize a list to store the average in-subject distances
average_in_subject_distances = []

# Iterate over the 8 sessions with multiple examples
for session in sessions_with_multiple_examples[:8]:
    # Iterate over the examples in the current session
    for i in range(len(session)):
        # Initialize a list to store the distances for the current example
        in_subject_distances = []
        # Compute the Euclidean distance to each other example within the same subject
        for j in range(len(session)):
            if i != j:  # Exclude the distance to the example itself
                distance = euclidean(session.iloc[i], session.iloc[j])
                in_subject_distances.append(distance)
        # Compute the average distance for the current example
        average_in_subject_distance = np.mean(in_subject_distances)
        # Add the average distance to the list
        average_in_subject_distances.append(average_in_subject_distance)

# Plot the average in-subject distances on the same plot, in red
plt.plot(average_in_subject_distances, color='red', label='In-Subject')

plt.title('Average Euclidean Distance between Examples')
plt.xlabel('Example')
plt.ylabel('Average Euclidean Distance')
plt.legend()
plt.show()