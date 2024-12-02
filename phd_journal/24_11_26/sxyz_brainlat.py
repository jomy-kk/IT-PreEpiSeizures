# Open electrode locs with pandas
filepath_locs = "/Volumes/MMIS-Saraiv/Datasets/BrainLat/electrode_locs.csv"
filepath_conversion = "/Volumes/MMIS-Saraiv/Datasets/BrainLat/denoised_fixed/electrode_conversion.csv"

import pandas as pd
locs = pd.read_csv(filepath_locs, sep=',', header=0, index_col=0)
#locs.index = locs.index.astype(str).str.strip()
conversion = pd.read_csv(filepath_conversion, sep=';', header=0, index_col=1)

channel_order = ('Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'T3', 'T4', 'C3', 'Cz', 'C4', 'T5', 'P3', 'Pz', 'P4', 'T6', 'O1', 'O2')

# Save to txt
# first row is the number of electrodes
# subsequent rows are the x, y, z, name coordinates of each electrode, separated by a space
with open("/Volumes/MMIS-Saraiv/Datasets/BrainLat/electrodes.sxyz", 'w', encoding='ascii') as f:
    f.write(f"{len(channel_order)}\n")
    for channel_name in channel_order:
        channel_name_biosemi = conversion.loc[channel_name]['Biosemi 128']
        row = locs.loc[channel_name_biosemi]
        f.write(f"{row['x']} {row['y']} {row['z']} {channel_name}\n")