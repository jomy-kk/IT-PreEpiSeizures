import os
from datetime import timedelta
from glob import glob
from os.path import join, exists, split

from ltbio.biosignals.modalities import EEG
from ltbio.biosignals.sources.Medipol import Medipol

common_path = '/Volumes/MMIS-Saraiv/Datasets/Istambul/denoised'
out_common_path = '/Volumes/MMIS-Saraiv/Datasets/Istambul/denoised_biosignal'
source = Medipol('/Volumes/MMIS-Saraiv/Datasets/Istambul/metadata.csv')

# Get recursively all subdirectories in common_path
all_subjects = glob(join(common_path, '*'), recursive=False)

for subject_path in all_subjects:
    subject = split(subject_path)[-1]
    subject_out_path = join(out_common_path, subject)

    if not exists(subject_out_path + '.biosignal'): # If the file does not exist yet, save it

        # Make Biosignal object
        try:
            res = EEG(subject_path, source)
            res.save(subject_out_path)
            print(f"Done {subject_path}.")

            # Take a sneak peek as well
            if res.duration >= timedelta(seconds=20):
                try:
                    res["O2"][:res.initial_datetime + timedelta(seconds=20)].plot(show=False, save_to=subject_out_path + '.png')
                except IndexError:
                    res["O2"][:res.domain[0].end_datetime].plot(show=False, save_to=subject_out_path + '.png')
                else:
                    res["O2"].plot(show=False, save_to=subject_out_path + '.png')

            print(res)
            print(res["O2"]._n_segments)
            print(res["O2"].domain)
            print()

        except FileNotFoundError as e:
            print(e)
