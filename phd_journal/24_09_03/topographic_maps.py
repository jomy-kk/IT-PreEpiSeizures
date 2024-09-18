import mne
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from mne.viz import plot_topomap
from matplotlib.colors import LinearSegmentedColormap
from pandas import Series

from read import read_all_features, read_ages
from utils import feature_wise_normalisation

def read_children(bad_diagnoses=True):
    # 1) Read features
    features = read_all_features('KJPP', multiples=True)
    features = features.dropna()  # drop sessions with missing values
    features.index = features.index.str.split('$').str[0]  # remove $ from the index

    # 2) Read targets
    kjpp_ages = read_ages('KJPP')
    targets = Series()
    for index in features.index:
        if '$' in str(index):  # Multiples
            key = str(index).split('$')[0]  # remove the multiple
        else:  # Original
            key = index
        if key in kjpp_ages:
            targets.loc[index] = kjpp_ages[key]

    targets = targets.dropna()  # Drop subject_sessions with nans targets
    features = features.loc[targets.index]
    print("Features Shape before drop wo/ages:", features.shape)

    # keep only ages <= 23
    targets = targets[targets <= 23]
    features = features.loc[targets.index]

    # 3) Normalisation feature-wise
    features = feature_wise_normalisation(features, method='min-max')

    # 4) Remove/Keep bad-diagnoses
    if not bad_diagnoses:
        BAD_DIAGNOSES = np.loadtxt("/Volumes/MMIS-Saraiv/Datasets/KJPP/session_ids/bad_diagnoses.txt", dtype=str)
        GOOD_DIAGNOSES = features.index.difference(BAD_DIAGNOSES)
        n_before = len(features)
        features = features.drop(BAD_DIAGNOSES, errors='ignore')
        targets = targets.drop(BAD_DIAGNOSES, errors='ignore')
        print("Removed Bad diagnoses:", n_before - len(features))
    else:
        # Separate bad-diagnoses to test set
        F8 = np.loadtxt("F8.txt", dtype=str)
        F7 = np.loadtxt("F7.txt", dtype=str)
        Q = np.loadtxt("Q.txt", dtype=str)
        BAD_DIAGNOSES = np.concatenate((F8, F7, Q))

        features = features[features.index.isin(BAD_DIAGNOSES)]
        targets = targets[targets.index.isin(BAD_DIAGNOSES)]
        print("Kept bad diagnoses:", len(features))

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
                     pos=info, sensors=True,names=channel_order, #show_names=False, # channel names and their order
                     mask=np.array([x in ('C3', 'C4', 'Cz', 'Fz') for x in channel_order]),  # channels to highlight
                     mask_params=dict(marker='o', markerfacecolor='w', markeredgecolor='k', linewidth=0, markersize=15, alpha=0.6),  # style of highlighted channels
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

        #plt.title("Age: {}-{}".format(group[0], group[1]))
        # no axes
        plt.axis('off')
        plt.grid(False)
        plt.tight_layout()
        plt.savefig("./topomaps/" + "topomap_{}_{}_{}_{}.png".format(feature_name, label, group[0], group[1]), dpi=300, bbox_inches='tight')


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
                     pos=info, sensors=True,names=channel_order, #show_names=True, # channel names and their order
                     mask=np.array([x in ('T4',) for x in channel_order]),  # channels to highlight
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

# Get Children
features_children, targets_children = read_children(bad_diagnoses=False)
features_children = features_children[feature_names]

#VMIN, VMAX = None, None
VMIN, VMAX = 0.23, 0.64

cmap = LinearSegmentedColormap.from_list("my_cmap", ["white", '#C60E4F'], N=256)
plot(features_children, targets_children, ((0, 8), (8, 10), (10, 12), (12,14), (14, 16), (16, 23)), label='age', vmin=VMIN, vmax=VMAX, cmap=cmap)
#plot(features_children, targets_children, [(a, a+1) for a in range(4, 19)], label='age', vmin=VMIN, vmax=VMAX, cmap=cmap)

plt.figure()
mappable = plt.cm.ScalarMappable(cmap=cmap)
mappable.set_clim(VMIN, VMAX)
plt.colorbar(mappable, shrink=0.5, aspect=20, orientation='horizontal')
plt.savefig("./topomaps/colorbar.png", dpi=300, bbox_inches='tight')
#"""

"""
# Make colourbar in horizontal
mappable = plt.cm.ScalarMappable(cmap='viridis')
mappable.set_clim(0, 0.4)
plt.colorbar(mappable, shrink=0.8, aspect=60, orientation='horizontal')
plt.savefig("/Users/saraiva/Desktop/Doktorand/Scientific Outputs/Journal Articles/RH-images/TopoMaps/colorbar.png", dpi=300, bbox_inches='tight')
"""