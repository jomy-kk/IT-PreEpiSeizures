import pandas as pd

from read import read_all_features, read_mmse

# 1) Read features
# 1.1. Multiples = yes
# 1.2. Which multiples = all
# 1.3. Which features = FEATURES_SELECTED
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
targets = pd.Series()
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

# 3) Statistics
print("Statistics")

insight_statistics, brainlat_statistics, miltiadous_statistics, spaienza_statistics = {}, {}, {}, {}
for index in features.index:

    session = index.split('$')[0]

    if '_' in str(index):  # insight
        if session in insight_statistics:
            insight_statistics[session] += 1
        else:
            insight_statistics[session] = 1

    elif '-' in str(index):  # brainlat
        if session in brainlat_statistics:
            brainlat_statistics[session] += 1
        else:
            brainlat_statistics[session] = 1

    elif 'PARTICIPANT' in str(index):  # sapienza
        if session in spaienza_statistics:
            spaienza_statistics[session] += 1
        else:
            spaienza_statistics[session] = 1

    else:  # miltiadous
        if session in miltiadous_statistics:
            miltiadous_statistics[session] += 1
        else:
            miltiadous_statistics[session] = 1

print("Mean # Examples per Unique Session")
print("INSIGHT:", sum(insight_statistics.values()) / len(insight_statistics))
print("BrainLat:", sum(brainlat_statistics.values()) / len(brainlat_statistics))
print("Sapienza:", sum(spaienza_statistics.values()) / len(spaienza_statistics))
print("Miltiadous:", sum(miltiadous_statistics.values()) / len(miltiadous_statistics))

print("Total Examples per Dataset")
print("INSIGHT:", sum(insight_statistics.values()))
print("BrainLat:", sum(brainlat_statistics.values()))
print("Sapienza:", sum(spaienza_statistics.values()))
print("Miltiadous:", sum(miltiadous_statistics.values()))
