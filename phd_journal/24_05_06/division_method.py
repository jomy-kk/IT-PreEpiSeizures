from glob import glob
from os.path import join

import pandas as pd
import networkx as nx
from sklearn.cluster import KMeans

# FEATURES
FEATURES_SELECTED_OLD = ['Spectral#RelativePower#C3#beta1', 'Spectral#EdgeFrequency#C3#beta3', 'Spectral#RelativePower#C3#gamma', 'Spectral#EdgeFrequency#C4#alpha1', 'Spectral#RelativePower#C4#beta3', 'Spectral#EdgeFrequency#C4#beta3', 'Spectral#EdgeFrequency#C4#gamma', 'Spectral#Flatness#Cz#theta', 'Spectral#PeakFrequency#Cz#theta', 'Spectral#EdgeFrequency#Cz#beta3', 'Spectral#EdgeFrequency#Cz#gamma', 'Spectral#PeakFrequency#Cz#gamma', 'Spectral#RelativePower#F3#beta1', 'Spectral#Diff#F4#delta', 'Spectral#RelativePower#F7#beta3', 'Spectral#EdgeFrequency#F7#beta3', 'Spectral#RelativePower#F7#gamma', 'Spectral#RelativePower#F8#beta1', 'Spectral#EdgeFrequency#F8#beta3', 'Spectral#RelativePower#Fp1#beta1', 'Spectral#EdgeFrequency#Fp1#beta3', 'Spectral#Diff#Fp2#delta', 'Spectral#RelativePower#Fp2#beta1', 'Spectral#RelativePower#Fp2#beta3', 'Spectral#Diff#Fpz#beta2', 'Spectral#Entropy#O1#delta', 'Spectral#RelativePower#O1#beta2', 'Spectral#EdgeFrequency#O1#beta2', 'Spectral#EdgeFrequency#O1#beta3', 'Spectral#RelativePower#O2#delta', 'Spectral#PeakFrequency#O2#alpha1', 'Spectral#RelativePower#O2#beta1', 'Spectral#RelativePower#O2#beta3', 'Spectral#Diff#P3#beta1', 'Spectral#RelativePower#P3#beta3', 'Spectral#RelativePower#Pz#alpha1', 'Spectral#EdgeFrequency#Pz#beta3', 'Spectral#RelativePower#T4#alpha1', 'Spectral#RelativePower#T4#beta3', 'Spectral#RelativePower#T4#gamma', 'Spectral#EdgeFrequency#T5#beta2', 'Hjorth#Complexity#T5', 'Hjorth#Complexity#P4', 'Hjorth#Complexity#F7', 'Hjorth#Complexity#T4', 'Hjorth#Complexity#F8', 'Hjorth#Complexity#T3', 'Hjorth#Mobility#P3', 'PLI#Frontal(L)-Temporal(R)#alpha1', 'PLI#Frontal(L)-Occipital(L)#alpha1', 'PLI#Frontal(R)-Temporal(R)#alpha1', 'PLI#Temporal(R)-Parietal(R)#alpha1', 'PLI#Temporal(R)-Occipital(L)#alpha1', 'PLI#Parietal(R)-Occipital(L)#alpha1', 'PLI#Occipital(L)-Occipital(R)#alpha1', 'PLI#Temporal(R)-Occipital(R)#alpha2', 'PLI#Parietal(R)-Occipital(L)#alpha2', 'COH#Frontal(L)-Frontal(R)#theta', 'COH#Frontal(L)-Occipital(L)#theta', 'COH#Frontal(L)-Occipital(R)#alpha1', 'COH#Frontal(R)-Occipital(L)#alpha1', 'COH#Parietal(R)-Occipital(L)#alpha1', 'COH#Frontal(L)-Frontal(R)#alpha2', 'COH#Frontal(L)-Occipital(R)#alpha2', 'COH#Parietal(R)-Occipital(L)#alpha2', 'COH#Parietal(R)-Occipital(R)#alpha2', 'COH#Occipital(L)-Occipital(R)#alpha2', 'COH#Frontal(L)-Occipital(L)#beta1', 'COH#Temporal(R)-Parietal(R)#beta1', 'COH#Parietal(R)-Occipital(R)#beta1', 'COH#Frontal(L)-Parietal(L)#beta2', 'COH#Frontal(R)-Occipital(L)#beta2', 'COH#Frontal(L)-Temporal(R)#beta3', 'COH#Frontal(L)-Parietal(L)#beta3', 'COH#Frontal(L)-Occipital(L)#beta3', 'COH#Frontal(L)-Occipital(R)#beta3', 'COH#Frontal(R)-Occipital(L)#beta3', 'COH#Temporal(L)-Occipital(R)#beta3', 'COH#Frontal(L)-Occipital(R)#gamma', 'COH#Frontal(R)-Occipital(R)#gamma']
FEATURES_SELECTED = []
for feature in FEATURES_SELECTED_OLD:
    if 'alpha1' in feature or 'alpha2' in feature or 'beta1' in feature or 'beta2' in feature or 'beta3' in feature:
        feature = feature[:-1]
    FEATURES_SELECTED.append(feature)
FEATURES_SELECTED = list(set(FEATURES_SELECTED))

dataset_path = '/Volumes/MMIS-Saraiv/Datasets/BrainLat/features'
#dataset_path = '/Volumes/MMIS-Saraiv/Datasets/Sapienza/features'
#dataset_path = '/Volumes/MMIS-Saraiv/Datasets/Miltiadous Dataset/features'

# Read all cohort files "*$Multiple.csv"
all_multiple_files = glob(join(dataset_path, 'Cohort*$Multiple.csv'))

# Read all DataFrames and concatenate their columns
all_features = [pd.read_csv(file, index_col=0) for file in all_multiple_files]
all_features = pd.concat(all_features, axis=1)
all_features = all_features[FEATURES_SELECTED]
all_features = all_features.dropna()

# Standardize features to have zero mean and unit variance
all_features = (all_features - all_features.mean()) / all_features.std()

# Indexes are: 'subject$multiple'
# Make two indices: 'subject' and 'multiple'
all_features['subject'] = all_features.index.str.split('$').str[0]
all_features['multiple'] = all_features.index.str.split('$').str[1]
all_features = all_features.set_index(['subject', 'multiple'])

independent_tuples_by_subject = {}
# Iterate by subject
for subject in all_features.index.get_level_values('subject').unique():
    print("##################")
    print("Subject: ", subject)
    subject_features = all_features.loc[subject]

    # Print the number of rows
    print("Number of rows:", len(subject_features))

    # STATISTICAL TESTS
    independent_rows = []
    dependent_rows = []
    from scipy.stats import mannwhitneyu
    for i in range(len(subject_features)):
        for j in range(i, len(subject_features)):
            # Perform <Mann-Whitney U test>
            u_stat, p = mannwhitneyu(subject_features.iloc[i], subject_features.iloc[j])
            # interpret
            # p > alpha: Likely come from the same distribution (fail to reject H0)
            # p <= alpha: Likely come from different distributions (reject H0)
            alpha = 0.05
            if p > alpha:
                #print(f'Rows ({i}, {j}): Likely come from the same subject (fail to reject H0). u_stat={u_stat}, p={p:.3f}')
                if i != j:
                    dependent_rows.append((i, j))
            else:
                print(f'Rows ({i}, {j}): Likely come from different subjects (reject H0) u_stat={u_stat}, p={p:.3f}')
                independent_rows.append((i, j))

    print("Number of independent pairs:", len(independent_rows))

    # print("##################")

    G = nx.Graph()
    # Add a node for each row
    for i in range(len(subject_features)):
        G.add_node(i)
    # Add an edge for each pair with p-value less than 0.05
    for i, j in independent_rows:
        G.add_edge(i, j)
    # Find the largest clique
    cliques = list(nx.find_cliques(G))
    print("All cliques (with size > 1):")
    for clique in cliques:
        if len(clique) > 1:
            print(clique)
    largest_clique = max(cliques, key=len)
    # print("The largest set of rows where all pairs have p<0.05 is:")
    # print(largest_clique)
    K = len(largest_clique)
    print(f'Length of the largest clique, {K}, will be K.')
    if K < 2:
        print("No independent pairs found.")
        independent_tuples_by_subject[subject] = []
        continue

    # Use a clustering algorithm
    K = len(largest_clique)
    kmeans = KMeans(n_clusters=K, random_state=0).fit(subject_features)
    labels = kmeans.labels_
    print(f'Cluster labels: {labels}')

    # Check if any of the independent pairs are in the same cluster
    to_exclude = []
    for i, j in independent_rows:
        if labels[i] == labels[j]:
            # print(f'Rows ({i}, {j}) are independent but in the same cluster')
            to_exclude.append((i, j))

    print("Independent pairs that can be safely used because they're not clustered together:")
    safe_pairs = []
    for i, j in independent_rows:
        if (i, j) not in to_exclude and (j, i) not in to_exclude:
            safe_pairs.append((i, j))
            print(f'Pair ({i}, {j})')

    cliques_K = [clique for clique in cliques if len(clique) == K]
    print(f"Cliques with {K} elements with safe pairs to use:")
    res = []
    for clique in cliques_K:
        if all((i, j) in safe_pairs or (j, i) in safe_pairs for i in clique for j in clique if i != j):
            print(clique)
            res.append(tuple(clique))

    independent_tuples_by_subject[subject] = str(res)
    pass

# Save the independent tuples by subject
independent_tuples_by_subject = pd.Series(independent_tuples_by_subject)
independent_tuples_by_subject.to_csv(join(dataset_path, 'safe_multiples.csv'))