from datetime import timedelta
from glob import glob
from os.path import join, exists, split

from ltbio.biosignals.modalities import EEG
from ltbio.biosignals.sources.BrainLat import BrainLat
from ltbio.biosignals.sources.KJPP import KJPP

common_path = '/Volumes/MMIS-Saraiv/Datasets/BrainLat/denoised_fixed'
out_common_path = '/Volumes/MMIS-Saraiv/Datasets/BrainLat/denoised_biosignal'
socio_demog = '/Volumes/MMIS-Saraiv/Datasets/BrainLat/metadata.csv'
source = BrainLat(socio_demog)

# Get recursively all .set files in common_path
all_session_directories = glob(join(common_path, '*.set'))
all_session_directories = sorted(all_session_directories)

for session_directory in all_session_directories:
    filename = split(session_directory)[-1]

    # Make Biosignal object
    try:
        x = EEG(session_directory, source)
    except LookupError as e:
        print(f"No age for {filename}: {e}")
        continue
    except FileNotFoundError:
        print(f"No files for {filename}.")
        continue
    except ValueError as e:
        print(e)
        continue
    # Structure its name
    short_patient_code = x.patient_code
    out_filename = short_patient_code
    out_filepath = join(out_common_path, out_filename + '.biosignal')

    # If the file does not exist yet, save it
    if not exists(out_filepath):
        print(f"Done {filename}.")
        x.save(out_filepath)
        # Take a sneak peek as well
        if x.duration >= timedelta(seconds=30):
            try:
                x["T5"][:x.initial_datetime+timedelta(seconds=30)].plot(show=False, save_to=join(out_common_path, out_filename + '.png'))
            except IndexError:
                x["T5"][:x.domain[0].end_datetime].plot(show=False, save_to=join(out_common_path, out_filename + '.png'))
        else:
            x["T5"].plot(show=False, save_to=join(out_common_path, out_filename + '.png'))
    # Delete the object to free memory
    del x

