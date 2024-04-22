from datetime import timedelta
from glob import glob
from os.path import join, exists, split

from ltbio.biosignals.modalities import EEG
from ltbio.biosignals.sources.KJPP import KJPP

common_path = '/Volumes/MMIS-Saraiv/Datasets/KJPP/autopreprocessed/1'
#common_path = '/Volumes/MMIS-Saraiv/Datasets/KJPP/test'
out_common_path = '/Volumes/MMIS-Saraiv/Datasets/KJPP/autopreprocessed_biosignal/1'
# socio_demog = '/Volumes/MMIS-Saraiv/Datasets/KJPP/demographics.csv'  # For future, when the demographics file is updated
socio_demog = '/Volumes/MMIS-Saraiv/Datasets/KJPP/metadata.csv'
source = KJPP(socio_demog)

# Get recursively all directories in common_path
all_session_directories = glob(join(common_path, '**/*'))

for session_directory in all_session_directories:
    session_code = split(session_directory)[-1]
    out_filename = session_code
    out_filepath = join(out_common_path, out_filename + '.biosignal')

    if not exists(out_filepath):

        # Make Biosignal object
        try:
            x = EEG(session_directory, source)
        except LookupError as e:
            print(e)
            continue
        except FileNotFoundError as e:
            print(e)
            continue
        # Structure its name
        short_patient_code = x.patient_code
        short_session_code = x.name
        # out_filename = short_patient_code + '_' + short_session_code # For future, when the demographics file is updated
        #out_filename = short_session_code
        #out_filepath = join(out_common_path, out_filename + '.biosignal')

        # If the file does not exist yet, save it
        print(f"Done {session_code}.")
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

