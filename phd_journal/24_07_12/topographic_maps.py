import mne
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from mne.viz import plot_topomap
from matplotlib.colors import LinearSegmentedColormap

from read import read_all_features, read_mmse, read_ages, read_diagnosis
from utils import feature_wise_normalisation, ELDERS_COLOUR, CHILDREN_COLOUR


def read_elders():
    # 1) Read features
    # 1.1. Multiples = yes
    # 1.2. Which multiples = all
    # 1.3. Which features = all
    miltiadous = read_all_features('Miltiadous Dataset', multiples=True)
    brainlat = read_all_features('BrainLat', multiples=True)
    sapienza = read_all_features('Sapienza', multiples=True)
    insight = read_all_features('INSIGHT', multiples=True)
    features = pd.concat([brainlat, miltiadous, sapienza, insight], axis=0)
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

    # 3) Read Diagnosis
    insight_targets = read_diagnosis('INSIGHT')
    brainlat_targets = read_diagnosis('BrainLat')
    miltiadous_targets = read_diagnosis('Miltiadous Dataset')
    sapienza_targets = read_diagnosis('Sapienza')
    diagnosis = pd.Series()
    batch = []
    for index in features.index:
        if '$' in str(index):  # Multiples
            key = str(index).split('$')[0]  # remove the multiple
        else:  # Original
            key = index

        if '_' in str(key):  # insight
            key = int(key.split('_')[0])
            if key in insight_targets:
                diagnosis.loc[index] = insight_targets[key]
                batch.append(1)
        elif '-' in str(key):  # brainlat
            if key in brainlat_targets:
                diagnosis.loc[index] = brainlat_targets[key]
                batch.append(2)
        elif 'PARTICIPANT' in str(key):  # sapienza
            if key in sapienza_targets:
                diagnosis.loc[index] = sapienza_targets[key]
                batch.append(3)
        else:  # miltiadous
            # parse e.g. 24 -> 'sub-024'; 1 -> 'sub-001'
            key = 'sub-' + str(key).zfill(3)
            if key:
                diagnosis.loc[index] = miltiadous_targets[key]
                batch.append(4)
    diagnosis = diagnosis.dropna()  # Drop subject_sessions with nans targets
    features = features.loc[diagnosis.index]

    # 4) Normalize features min-max
    #features = (features - features.min()) / (features.max() - features.min())

    return features, targets, diagnosis


def read_children(features_elders, targets_elders):
    # 1) Read features
    # 1.1. Multiples = yes
    # 1.3. Which features = FEATURES_SELECTED
    features = read_all_features('KJPP', multiples=True)
    features.index = features.index.str.split('$').str[0]  # remove $ from the index

    # 1.2.1) Remove the ones with bad-diagnoses
    BAD_DIAGNOSES = np.loadtxt("/Volumes/MMIS-Saraiv/Datasets/KJPP/session_ids/bad_diagnoses.txt", dtype=str)
    n_before = len(features)
    features = features.drop(BAD_DIAGNOSES, errors='ignore')
    print("Removed Bad diagnoses:", n_before - len(features))

    # 1.2.2) Remove others
    # 1.2.2) Remove others
    REMOVED_SESSIONS = np.loadtxt("/Users/saraiva/PycharmProjects/LTBio/phd_journal/24_03_18/inverse_problem3/scheme57/removed_sessions.txt", dtype=str)
    n_before = len(features)
    features = features.drop(REMOVED_SESSIONS, errors='ignore')
    print("Removed:", n_before - len(features))

    # 2) Get targerts
    targets = pd.Series()
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

    # Select only the features that are in the elders dataset
    features = features[features_elders.columns]

    # Remove ages >20
    features = features[targets <= 20]
    targets = targets[targets <= 20]

    # 3) Normalisation
    # 3.1. Normalisation method = min-max
    #elders_max = features_elders.max()
    #elders_min = features_elders.min()
    #features = (features - elders_min) / (elders_max - elders_min)
    #features = feature_wise_normalisation(features, 'min-max')

    # 4) Calibration
    # 4.4. method = by groups
    """
    #a, b, c, d = (0, 9), (9, 15), (15, 24), (24, 30)  # Elderly groups
    a, b, c, d = (0, 15), (16, 25), (26, 29), (29, 31)  # NEW Elderly groups
    #alpha, beta, gamma, delta = (0, 5), (5, 8), (8, 13), (13, 25)  # Children groups
    alpha, beta, gamma, delta = (0, 4.5), (4, 6), (6, 12), (12, 20)  # NEW Children groups

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
    """
    return features, targets


########

# Channels Info
channel_order = ['C3', 'C4', 'Cz', 'F3', 'F4', 'F7', 'F8', 'Fp1', 'Fp2', 'Fpz', 'Fz', 'O1', 'O2', 'P3', 'P4', 'Pz',
                 'T3', 'T4', 'T5', 'T6']
# Create an Info object, necessary for creating Evoked object
info = mne.create_info(ch_names=channel_order, sfreq=1, ch_types='eeg')
montage = mne.channels.make_standard_montage('standard_1020')
info.set_montage(montage)

def plot(F, T, groups, label, vmin=None, vmax=None, cmap=None):

    if vmin is None:
        vmin = F.min().min()
    if vmax is None:
        vmax = F.max().max()

    print("Vmin:", vmin)
    print("Vmax:", vmax)

    for i, group in enumerate(groups):
        # Filter by target
        F_group = F.loc[T.between(*group)]

        print("Group:", group)
        print("Number of examples:", F_group.shape[0])

        # Keep only one session per subject and discard remainder
        F_group = F_group.groupby(F_group.index.str.split('$').str[0]).mean()

        # Average all sessions
        F_group = F_group.mean(axis=0).values

        print("Max:", F_group.max())
        print("Min:", F_group.min())

        print("Average:", F_group.mean())

        # Make dimensions compatible
        F_group = np.array(F_group)

        # arial 12pt
        plt.rc('font', size=10)
        plt.rc('font', family='Arial')

        # Plot the topomap
        plot_topomap(F_group, vlim=(vmin, vmax),  # features and their range
                     pos=info, sensors=True,names=None, #channel_order, #show_names=False, # channel names and their order
                     #mask=np.array([x in ('T5', 'F4', 'T3', 'P3', 'O2', 'Cz') for x in channel_order]),  # channels to highlight
                     #mask=np.array([x in ('T4', 'T3', 'Cz') for x in channel_order]),  # channels to highlight
                     #mask=np.array([x in ('O2', 'O1', 'Fz') for x in channel_order]),  # channels to highlight
                     #mask=np.array([x in ('O1', 'P4',) for x in channel_order]),  # channels to highlight
                     #mask=np.array([x in ('T3',) for x in channel_order]),  # channels to highlight
                     #mask_params=dict(marker='o', markerfacecolor='w', markeredgecolor='k', linewidth=0, markersize=15, alpha=0.6),  # style of highlighted channels
                     cmap='viridis' if cmap is None else cmap, #colorbar=True,  # colormap
                     outlines='head', contours=6, image_interp='cubic', border='mean',  # head shape and interpolation
                     axes=None, res=1024, show=False, size=3)  # resolution and size

        """
        # Annotate the topomap with the amplitude values
        for ch_name, ch_val in zip(channel_order, F_group):
            if ch_name in ('T3',):  # channels to highlight
                x, y = info['chs'][channel_order.index(ch_name)]['loc'][:2]
                plt.annotate(f"{ch_val:.2f}", xy=(x-0.007, y-0.015), xytext=(0, 10), textcoords='offset points', ha='center',
                             va='bottom', color='black')
        """

        # no axes
        plt.axis('off')
        plt.grid(False)
        plt.tight_layout()
        plt.savefig("/Users/saraiva/Desktop/Doktorand/Scientific Outputs/Journal Articles/RH-images/TopoMaps/" + "topomap_{}_{}_{}_{}.png".format(feature_name, label, group[0], group[1]), dpi=300, bbox_inches='tight')


def plot_by_stage(F, T, D, vmin=None, vmax=None, cmap=None):

    if vmin is None:
        vmin = F.min().min()
    if vmax is None:
        vmax = F.max().max()

    print("Vmin:", vmin)
    print("Vmax:", vmax)

    # Plot 1: (HC or SMC) and MMSE == 30
    # Plot 2: HC and MMSE < 30
    # Plot 3: SMC and MMSE < 30
    # Plot 4: AD and MMSE > 13
    # Plot 5: AD abd MMSE <= 13
    # Plot 6: FTD and MMSE > 24
    # Plot 7: FTD and MMSE <= 24

    groups = (
        (("HC", "SMC"), (30, 30),),
        (("HC", ), (0, 29),),
        (("SMC", ), (0, 29),),
        (("AD", ), (13, 30),),
        (("AD", ), (0, 13),),
        (("FTD", ), (24, 30),),
        (("FTD", ), (0, 24),),
    )

    for i, (d, mmse) in enumerate(groups):
        # Filter by diagnosis
        if len(d) == 1:
            F_group = F.loc[D == d[0]]
        elif len(d) == 2:
            F_group = F.loc[(D == d[0]) | (D == d[1])]
        else:
            raise ValueError("Only two diagnosis are supported")

        # Filter by target
        F_group = F_group.loc[T.between(*mmse)]

        print("Diagnosis:", d)
        print("MMSE rage:", mmse)
        print("Number of examples:", F_group.shape[0])

        # Keep only one session per subject and discard remainder
        F_group = F_group.groupby(F_group.index.str.split('$').str[0]).mean()

        # Average all sessions
        F_group = F_group.mean(axis=0).values

        print("Max:", F_group.max())
        print("Min:", F_group.min())
        print("Average:", F_group.mean())

        # Make dimensions compatible
        F_group = np.array(F_group)

        # arial 12pt
        plt.rc('font', size=10)
        plt.rc('font', family='Arial')

        # Plot the topomap
        plot_topomap(F_group, vlim=(vmin, vmax),  # features and their range
                     pos=info, sensors=True,names=None, #channel_order, #show_names=False, # channel names and their order
                     #mask=np.array([x in ('T5', 'F4', 'T3', 'P3', 'O2', 'Cz') for x in channel_order]),  # channels to highlight
                     #mask=np.array([x in ('T4', 'T3', 'Cz') for x in channel_order]),  # channels to highlight
                     #mask=np.array([x in ('O2', 'O1', 'Fz') for x in channel_order]),  # channels to highlight
                     #mask=np.array([x in ('O1', 'P4',) for x in channel_order]),  # channels to highlight
                     mask=np.array([x in ('P4',) for x in channel_order]),  # channels to highlight
                     mask_params=dict(marker='o', markerfacecolor='w', markeredgecolor='k', linewidth=0, markersize=15, alpha=0.6),  # style of highlighted channels
                     cmap='viridis' if cmap is None else cmap, #colorbar=True,  # colormap
                     outlines='head', contours=6, image_interp='cubic', border='mean',  # head shape and interpolation
                     axes=None, res=1024, show=False, size=3)  # resolution and size

        # no axes
        plt.axis('off')
        plt.grid(False)
        plt.tight_layout()
        plt.savefig("/Users/saraiva/Desktop/Doktorand/Scientific Outputs/Journal Articles/RH-images/TopoMaps/" + "topomap_{}_{}_{}_{}.png".format(feature_name, d, mmse[0], mmse[1]), dpi=300, bbox_inches='tight')



#######

#"""
feature_name = 'Hjorth#Mobility'
feature_names = ['{}#{}'.format(feature_name, channel) for channel in channel_order]

# Get Elders
features_elders, targets_elders, diagnosis = read_elders()
features_elders = features_elders[feature_names]

# Stochastic pattern of elders
#elders_stochastic_pattern = pd.DataFrame([features_elders.min(), features_elders.max(), features_elders.mean(), features_elders.std()], index=['min', 'max', 'mean', 'std'])
#elders_stochastic_pattern.to_csv('elders_stochastic_pattern.csv')

# Get Children
features_children, targets_children = read_children(features_elders, targets_elders)
features_children = features_children[feature_names]

# Normalise children features
#features_children = feature_wise_normalisation(features_children, 'min-max')

VMIN, VMAX = 8.5, 10.5

# Plot

cmap = LinearSegmentedColormap.from_list("my_cmap", ["white", ELDERS_COLOUR], N=256)

plot_by_stage(features_elders, targets_elders, diagnosis, vmin=VMIN, vmax=VMAX, cmap=cmap)
#plot(features_elders, targets_elders, ((0, 12), (13, 23), (24, 29), (30, 30)), label='mmse', vmin=0.1, vmax=0.42, cmap=cmap)
#plot(features_elders, targets_elders, ((0, 15), (16, 25), (26, 28), (27, 29), (28, 30)), label='mmse', vmin=0, vmax=0.4)
#plot(features_elders, targets_elders, ((0, 18), (18, 28) ), label='mmse', vmin=0, vmax=0.4)

plt.figure()
mappable = plt.cm.ScalarMappable(cmap=cmap)
mappable.set_clim(VMIN, VMAX)
plt.colorbar(mappable, shrink=0.5, aspect=20, orientation='vertical')
plt.savefig("/Users/saraiva/Desktop/Doktorand/Scientific Outputs/Journal Articles/RH-images/TopoMaps/colorbar_edlers.png", dpi=300, bbox_inches='tight')

cmap = LinearSegmentedColormap.from_list("my_cmap", ["white", CHILDREN_COLOUR], N=256)
plot(features_children, targets_children, ((0, 8), (8, 12), (12, 17), (17, 20)), label='age', vmin=VMIN, vmax=VMAX, cmap=cmap)
#plot(features_children, targets_children, ((0, 4.5), (4.5, 6), (6, 8), (8, 12), (12, 19)), label='age', vmin=0, vmax=0.4)
#plot(features_children, targets_children, ((0, 4.7), ), label='age', vmin=0, vmax=0.4)
plt.figure()
mappable = plt.cm.ScalarMappable(cmap=cmap)
mappable.set_clim(VMIN, VMAX)
plt.colorbar(mappable, shrink=0.5, aspect=20, orientation='vertical')
plt.savefig("/Users/saraiva/Desktop/Doktorand/Scientific Outputs/Journal Articles/RH-images/TopoMaps/colorbar_children.png", dpi=300, bbox_inches='tight')
#"""

"""
# Make colourbar in horizontal
mappable = plt.cm.ScalarMappable(cmap='viridis')
mappable.set_clim(0, 0.4)
plt.colorbar(mappable, shrink=0.8, aspect=60, orientation='horizontal')
plt.savefig("/Users/saraiva/Desktop/Doktorand/Scientific Outputs/Journal Articles/RH-images/TopoMaps/colorbar.png", dpi=300, bbox_inches='tight')
"""