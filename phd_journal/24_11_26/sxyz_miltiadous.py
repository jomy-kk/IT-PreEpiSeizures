# Open electrode locs with pandas
filepaths = "/Volumes/MMIS-Saraiv/Datasets/Miltiadous Dataset/channel_locs.xyz"

import pandas as pd
locs = pd.read_csv(filepaths, sep='\t', header=None, names=['number', 'x', 'y', 'z', 'name'], index_col=4)
locs.index = locs.index.astype(str).str.strip()

channel_order = ('Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'T3', 'T4', 'C3', 'Cz', 'C4', 'T5', 'P3', 'Pz', 'P4', 'T6', 'O1', 'O2')

# Save to txt
# first row is the number of electrodes
# subsequent rows are the x, y, z, name coordinates of each electrode, separated by a space
with open("/Volumes/MMIS-Saraiv/Datasets/Miltiadous Dataset/electrodes.sxyz", 'w', encoding='ascii') as f:
    f.write(f"{len(channel_order)}\n")
    for channel_name in channel_order:
        row = locs.loc[channel_name]
        f.write(f"{row['x']} {row['y']} {row['z']} {channel_name}\n")