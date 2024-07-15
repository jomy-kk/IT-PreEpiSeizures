from glob import glob
from os.path import join

import pandas as pd
import networkx as nx
from sklearn.cluster import KMeans
from scipy.stats import mannwhitneyu
from scipy.stats import ks_2samp

# FEATURES
FEATURES_SELECTED = ['Hjorth#Complexity#T5', 'Hjorth#Complexity#F4',
                     'COH#Frontal(R)-Parietal(L)#delta', 'Hjorth#Complexity#T3',
                     'Spectral#RelativePower#F7#theta', 'COH#Frontal(R)-Temporal(L)#theta',
                     'Spectral#EdgeFrequency#O2#beta', 'COH#Frontal(L)-Temporal(R)#beta',
                     'COH#Temporal(L)-Parietal(L)#gamma', 'Spectral#EdgeFrequency#O1#beta',
                     'COH#Frontal(R)-Parietal(L)#theta', 'COH#Temporal(L)-Temporal(R)#alpha',
                     'COH#Frontal(R)-Temporal(L)#gamma', 'COH#Temporal(R)-Parietal(L)#beta',
                     'COH#Frontal(R)-Occipital(L)#theta', 'COH#Temporal(L)-Parietal(L)#beta',
                     'Hjorth#Activity#F7', 'COH#Occipital(L)-Occipital(R)#gamma',
                     'Spectral#Flatness#P3#beta', 'COH#Temporal(R)-Parietal(R)#alpha',
                     'Spectral#Entropy#P3#alpha', 'COH#Frontal(R)-Parietal(R)#theta',
                     'COH#Frontal(R)-Temporal(L)#delta', 'Spectral#Entropy#O2#alpha',
                     'Spectral#Entropy#T4#theta', 'Spectral#RelativePower#Cz#beta',
                     'Spectral#Diff#Pz#delta', 'COH#Parietal(R)-Occipital(L)#beta',
                     'Spectral#EdgeFrequency#Fz#beta', 'Spectral#Diff#Cz#gamma',
                     'Spectral#RelativePower#Fp1#gamma', 'COH#Frontal(R)-Parietal(L)#gamma',
                     'PLI#Frontal(R)-Parietal(L)#alpha', 'Spectral#Diff#F7#beta',
                     'Hjorth#Mobility#O1', 'Spectral#Flatness#T4#gamma',
                     'PLI#Parietal(L)-Occipital(L)#gamma', 'Spectral#Flatness#T6#delta',
                     'COH#Parietal(R)-Occipital(L)#alpha',
                     'COH#Parietal(R)-Occipital(R)#beta', 'Spectral#Diff#T4#delta',
                     'Spectral#Diff#F8#alpha', 'COH#Temporal(R)-Occipital(L)#beta',
                     'COH#Parietal(R)-Occipital(L)#gamma', 'Hjorth#Mobility#P4',
                     'COH#Frontal(L)-Temporal(L)#beta',
                     'COH#Occipital(L)-Occipital(R)#alpha', 'Spectral#Entropy#T3#theta',
                     'COH#Frontal(R)-Occipital(R)#alpha', 'Hjorth#Complexity#P3',
                     'COH#Frontal(L)-Occipital(L)#beta', 'Hjorth#Activity#C3',
                     'COH#Temporal(L)-Occipital(R)#theta', 'Spectral#Diff#F4#beta',
                     'COH#Frontal(L)-Frontal(R)#gamma', 'Spectral#Diff#C3#gamma',
                     'COH#Frontal(L)-Frontal(R)#theta', 'COH#Parietal(L)-Occipital(R)#theta',
                     'Spectral#RelativePower#F7#gamma', 'Spectral#RelativePower#F3#beta',
                     'PLI#Temporal(R)-Parietal(R)#beta', 'Spectral#Flatness#F7#beta',
                     'Hjorth#Complexity#O2', 'Spectral#Entropy#Cz#theta',
                     'PLI#Frontal(R)-Occipital(R)#beta', 'COH#Temporal(L)-Parietal(R)#beta',
                     'COH#Frontal(L)-Occipital(L)#delta', 'Spectral#Flatness#F8#delta',
                     'Spectral#Entropy#F4#delta', 'PLI#Temporal(R)-Parietal(R)#gamma',
                     'COH#Occipital(L)-Occipital(R)#delta',
                     'COH#Temporal(L)-Parietal(R)#delta', 'PLI#Frontal(L)-Temporal(R)#delta',
                     'Spectral#Flatness#P3#theta', 'Spectral#Entropy#F7#alpha',
                     'COH#Frontal(R)-Temporal(R)#delta', 'COH#Frontal(L)-Occipital(R)#gamma',
                     'COH#Frontal(L)-Frontal(R)#beta', 'Hjorth#Complexity#Cz',
                     'COH#Frontal(L)-Occipital(R)#beta']

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

    for i in range(len(subject_features)):
        for j in range(i, len(subject_features)):
            # Perform <Mann-Whitney U test>
            u_stat, p = ks_2samp(subject_features.iloc[i], subject_features.iloc[j])
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
independent_tuples_by_subject.to_csv(join(dataset_path, 'new_safe_multiples.csv'))