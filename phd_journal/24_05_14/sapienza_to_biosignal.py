from datetime import timedelta
from glob import glob
from os.path import join, exists, split

from ltbio.biosignals.modalities import EEG
from ltbio.biosignals.sources.Sapienza import Sapienza

common_path = '/Volumes/MMIS-Saraiv/Datasets/Sapienza/denoised'
out_common_path = '/Volumes/MMIS-Saraiv/Datasets/Sapienza/denoised_biosignal'
socio_demog = '/Volumes/MMIS-Saraiv/Datasets/Sapienza/metadata.csv'
source = Sapienza(socio_demog)

# Get recursively all .set files in common_path
all_files = glob(join(common_path, '**/*.set'))

for F in all_files:
    filename = split(F)[-1]
    print(F)

    # Make Biosignal object
    x = EEG(F, source)

    # Structure its name
    short_patient_code = x.patient_code
    out_filename = short_patient_code
    out_filepath = join(out_common_path, out_filename + '.biosignal')

    # If the file does not exist yet, save it
    if not exists(out_filepath):

        # Change names of channels from 10-10 to 10-20
        x.set_channel_name('T7', 'T3')
        x.set_channel_name('T8', 'T4')
        x.set_channel_name('P7', 'T5')
        x.set_channel_name('P8', 'T6')

        # Save
        x.save(out_filepath)
        print(f"Done {filename}.")

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

