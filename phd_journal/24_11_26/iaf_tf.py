from glob import glob
from os.path import join, basename

import numpy as np
import pandas as pd
from biosppy.signals.tools import welch_spectrum
from matplotlib import pyplot as plt
import seaborn as sns

#####################
# THIS IS SPECIFIC FOR 19 ELECTRODES
# e.g. Miltaidous Dataset, BrainLat Dataset
#####################

#common_path = "/Volumes/MMIS-Saraiv/Datasets/Miltiadous Dataset/denoised_txt_epochs"
common_path = "/Volumes/MMIS-Saraiv/Datasets/BrainLat/denoised_txt_epochs/CL"
#out_path = '/Volumes/MMIS-Saraiv/Datasets/Miltiadous Dataset/iaf_tf_my-boudaries.csv'
out_path = ('/Volumes/MMIS-Saraiv/Datasets/BrainLat/iaf_tf_my-boudaries_CL.csv')
sf = 128.0

all_subjects = glob(join(common_path, '*'))
res = pd.DataFrame(columns=['Subject', 'IAF', 'TF'])

MULTIPLIER = 2
for subject_path in all_subjects:
    subject = basename(subject_path)
    # subject = basename(subject_path).split('PARTICIPANT')[1]
    print("Subject Code", subject)
    all_files = glob(join(subject_path, '*.txt'))
    power = []
    freqs = None
    # go by epoch
    for file in all_files:
        data = pd.read_csv(file, header=None, sep=' ', dtype=float)  # time x channels
        # select electrodes O1 and O2
        freqs1, power1 = welch_spectrum(data.iloc[:, -1], sampling_rate=sf, decibel=True, size=int(sf * MULTIPLIER - 1))  # O2
        freqs2, power2 = welch_spectrum(data.iloc[:, -2], sampling_rate=sf, decibel=True, size=int(sf * MULTIPLIER - 1))  # O1
        # freqs3, power3 = welch_spectrum(data.iloc[:, 19], sampling_rate=sf, decibel=True, size=int(sf*MULTIPLIER-1))
        assert np.all(freqs1 == freqs2)  # and np.all(freqs2 == freqs3)
        # Average power of the three electrodes
        # power.append((power1 + power2 + power3) / 3)
        power.append((power1 + power2) / 3)
        freqs = freqs1

    # average power across epochs
    power_avg = np.mean(np.array(power), axis=0)

    # find IAF between 8 and 14 Hz
    iaf = freqs[np.argmax(power_avg[8 * MULTIPLIER:12 * MULTIPLIER]) + 8 * MULTIPLIER]
    iaf = round(iaf, 1)
    print("IAF", iaf)
    # find TF between 3 and 8 Hz
    tf = freqs[np.argmin(power_avg[4 * MULTIPLIER:8 * MULTIPLIER]) + 4 * MULTIPLIER]
    tf = round(tf, 1)
    print("TF", tf)

    # plot all psds
    sns.set_palette("husl")
    plt.figure(figsize=(10, 5))
    for p in power:
        sns.lineplot(x=freqs[:46], y=p[:46], alpha=0.05, color='orange')
    sns.lineplot(x=freqs[:46], y=power_avg[:46], linewidth=3, color='black')
    plt.axvline(iaf, color='red', linestyle='--')
    plt.axvline(tf, color='blue', linestyle='--')
    sns.despine()
    plt.title("Subject Code {}".format(subject))
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Power (dB)")
    plt.savefig(join(subject_path, 'psd.png'), dpi=300, bbox_inches='tight')


    res = res.append({'Subject': subject, 'IAF': iaf, 'TF': tf}, ignore_index=True)

res.index = res['Subject']
res = res.drop(columns=['Subject'])
res.to_csv(out_path)
