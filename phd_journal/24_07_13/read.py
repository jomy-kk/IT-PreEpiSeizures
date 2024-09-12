from os.path import join
from pandas import read_csv
from pickle import load

common_path = '/Users/saraiva/PycharmProjects/IT-LongTermBiosignals/phd_journal/24_03_18/inverse_problem3/scheme57'


def read_elders():
    features = read_csv(join(common_path, 'elders_features.csv'), index_col=0)
    targets = read_csv(join(common_path, 'elders_targets.csv'), index_col=0)
    targets = targets['0']  # targets to DataFrame -> Series
    return features, targets


def read_children():
    features = read_csv(join(common_path, 'children_features.csv'), index_col=0)
    targets = read_csv(join(common_path, 'children_targets.csv'), index_col=0)
    targets = targets['0']  # targets to DataFrame -> Series
    return features, targets


def load_model():
    return load(open(join(common_path, 'model.pkl'), 'rb'))

# Feature names
feature_names = ['Hjorth#Complexity#T5', 'Hjorth#Complexity#F4',
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

