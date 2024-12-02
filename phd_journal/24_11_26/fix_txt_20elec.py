from glob import glob
from os.path import join
import pandas as pd

#common_in_path = "/Volumes/MMIS-Saraiv/Datasets/Miltiadous Dataset/denoised_txt_epochs/"
common_in_path = "/Volumes/MMIS-Saraiv/Datasets/BrainLat/denoised_txt_epochs/"
all_files = glob(join(common_in_path, "**/*.txt"), recursive=True)

for file in all_files:
    # Load txt to pandas
    x = pd.read_csv(file, sep='\t', header=None)
    # Delete last column of nans
    x = x.iloc[:, :-1]
    # Save to txt again
    x.to_csv(file, sep=' ', header=False, index=False)
    print(f"Done {file}.")


