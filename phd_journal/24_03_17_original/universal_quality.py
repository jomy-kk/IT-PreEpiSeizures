from datetime import timedelta
from glob import glob
from os.path import join, exists

import numpy as np
from datetimerange import DateTimeRange

from ltbio.biosignals.modalities import EEG
from ltbio.biosignals.timeseries import Timeline
from ltbio.processing.formaters import Segmenter, Normalizer


# FIXME: Change common_path to applicable path
common_path = '/Volumes/MMIS-Saraiv/Datasets/KJPP/autopreprocessed_biosignal/2'

# Get recursively all .biosignal files in common_path
all_files = glob(join(common_path, '*.biosignal'), recursive=True)

for filepath in all_files:
    filename = filepath.split('/')[-1].split('.')[0]
    out_filepath = join(common_path, filename + "_good.timeline")
    sqis_filepath = join(common_path, filename + "_sqis.txt")

    if not exists(out_filepath):
        print(filename)

        # Load
        x = EEG.load(filepath)

        if len(x) == 0:
            print(f"Empty file: {filename}")
            continue

        # Normalize
        normalizer = Normalizer(method='mean')
        x = normalizer(x)

        # Traverse segments
        domain = x['T5'].domain
        good_timelines = []
        for i, segment_domain in enumerate(domain):
            segment = x[segment_domain.start_datetime:segment_domain.end_datetime]
            with open(sqis_filepath, 'a') as f:
                f.write("Segment: " + str(i + 1) + "\n")
            #if segment.duration < timedelta(seconds=5):  # discard
            #    print("Discarding for short duration:", segment.duration)
            #else:
            # Segment in windows of 2 seconds
            z = Segmenter(timedelta(seconds=2))(segment)
            windows = np.array(z['T5'].domain)
            # Compute SQI by window
            oha_sqi = np.array(z.oha_sqi(threshold=2, by_segment=True))
            thv_sqi = np.array(z.thv_sqi(threshold=2, by_segment=True))
            chv_sqi = np.array(z.chv_sqi(1.5, None, None, by_segment=True))
            # Print results
            with open(sqis_filepath, 'a') as f:
                f.write("OHA: " + str(oha_sqi) + "\n")
                f.write("THV: " + str(thv_sqi) + "\n")
                f.write("CHV: " + str(chv_sqi) + "\n")
            #print("OHA:", oha_sqi)
            #print("THV:", thv_sqi)
            #print("CHV:", chv_sqi)
            # Find good quality segments
            good = (oha_sqi < 0.1) & (thv_sqi < 0.1) & (chv_sqi < 0.15)
            good_windows = windows[good]
            # Union of these windows
            good_windows = [Timeline(Timeline.Group([DateTimeRange(w.start_datetime, w.end_datetime)])) for w in good_windows]
            if len(good_windows) == 0:
                continue
            elif len(good_windows) > 1:
                good_windows = Timeline.union(*good_windows)
            else:
                good_windows = good_windows[0]
            good_windows.name = f"Good quality period of segment {i}"
            #good_windows.plot()
            good_timelines.append(good_windows)

        # Union of all good quality periods
        if len(good_timelines) > 1:
            general_good = Timeline.union(*good_timelines)
        else:
            general_good = good_timelines[0]
        general_good.name = f"Good quality periods of {filename}"
        general_good.plot(show=False, save_to=join(common_path, filename + "_good.png"))
        general_good.save(out_filepath)

        with open(sqis_filepath, 'a') as f:
            before = x['T5'].duration
            after = general_good.duration
            f.write(f'{before} => {after} ({100 - (before/after*100)}% discarded) \n')
