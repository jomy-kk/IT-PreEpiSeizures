import os
from datetime import timedelta
from glob import glob
from os.path import join, exists, split

from ltbio.biosignals.modalities import EEG
from ltbio.biosignals.sources.HBN import HealthyBrainNetwork

common_path = '/Volumes/MMIS-Saraiv/Datasets/Healthy Brain Network/fixed_and_segmented'
out_common_path = '/Volumes/MMIS-Saraiv/Datasets/Healthy Brain Network/biosignal'
#socio_demog = '/Volumes/MMIS-Saraiv/Datasets/Miltiadous Dataset/participants.tsv'
source = HealthyBrainNetwork()

# Get recursively all subdirectories in common_path
all_subjects = glob(join(common_path, '*'), recursive=False)

for subject_path in all_subjects:
    subject = split(subject_path)[-1]
    subject_out_path = join(out_common_path, subject)

    if not exists(subject_out_path + '.biosignal'): # If the file does not exist yet, save it

        subject_biosignals = []
        duration = timedelta(seconds=0)

        # Get all .edf files in the subject directory
        all_files = glob(join(subject_path, '*.edf'), recursive=False)

        if len(all_files) > 0:

            for F in all_files:
                # Structure its name
                filename = split(F)[-1]
                # Make Biosignal object
                x = EEG(F, source)
                x.timeshift(duration)
                print(f"Done {filename}.")
                #x.save(out_filepath)
                subject_biosignals.append(x)

                duration += x.duration + timedelta(seconds=2)

                # Delete the object to free memory
                del x

            # Concatenate all biosignals
            res = subject_biosignals[0]
            for x in subject_biosignals[1:]:
                res = res >> x
            res.save(subject_out_path + '.biosignal')
            print("Done", subject)

            # Take a sneak peek as well
            if res.duration >= timedelta(seconds=20):
                try:
                    res["T5"][:res.initial_datetime + timedelta(seconds=20)].plot(show=False, save_to=subject_out_path + '.png')
                except IndexError:
                    res["T5"][:res.domain[0].end_datetime].plot(show=False, save_to=subject_out_path + '.png')
                else:
                    res["T5"].plot(show=False, save_to=subject_out_path + '.png')

