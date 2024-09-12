# -- encoding: utf-8 --
# ===================================
# Name: Greedy EEG-Trim
# Description: Automatic greedy selection of I (e.g. I=2) good/useful EEG segments from raw EDF files.
#
# Contributors: João Saraiva
# Created: 01.02.2024
# Last Updated: 01.02.2024
#
# Created at Universität Rostock, DZNE and Universitätmedizin Rostock.
# ===================================

import glob
import json
import sys
import warnings
from configparser import ConfigParser
from os import mkdir, devnull, sep as os_sep
from os.path import join, isdir, isfile
from typing import Union

import scipy.io as sio
import matplotlib
import mne
import numpy as np
from mne import Annotations

# CHANGE THE VARIABLES BELOW ON CONFIG.INI
COMMON_PATH: str
OUTPUT_PATH: str
TRIM_EDGES: bool
DISCARD_BEGIN: float
MIN_DURATION_GOOD: float
PLOT_INF_PADDING: float
PLOT_SUP_PADDING: float
VISUALLY_CONFIRM: bool
PLOT_EEG_SCALING: float
DISCARD_FILE: list[str]
ONLY_PAIRED_ANNOT: bool

# ===================================

# DO NOT CHANGE ANYTHING BELOW THIS LINE

SINGLE_ANNOTATIONS_PATH = 'annotations_singles.json'
PAIR_ANNOTATIONS_PATH = 'annotations_pairs.json'
PROCESSED_PATH = '/Volumes/MMIS-Saraiv/Datasets/Healthy Brain Network/fixed_and_segmented/processed.txt'

INTERRUPTION_LABEL = 'interruption'

eeg_channels = list(range(0, 20))
relevant_channels = eeg_channels # + [24, 25, 30, 31]

DELTA = 0.05  # seconds


def read_config_file():
    config = ConfigParser()
    config.read('config.ini')
    global COMMON_PATH, OUTPUT_PATH, TRIM_EDGES, DISCARD_BEGIN, DISCARD_END, MIN_DURATION_GOOD, PLOT_INF_PADDING, PLOT_SUP_PADDING, VISUALLY_CONFIRM, PLOT_EEG_SCALING, DISCARD_FILE, ONLY_PAIRED_ANNOT
    COMMON_PATH = config['ENVIRONMENT']['COMMON_PATH']
    OUTPUT_PATH = config['ENVIRONMENT']['OUTPUT_PATH']
    TRIM_EDGES = config['TRIMMING']['TRIM_EDGES'].lower() == 'true'
    DISCARD_BEGIN = float(config['TRIMMING']['DISCARD_BEGIN'])
    DISCARD_END = float(config['TRIMMING']['DISCARD_END'])
    MIN_DURATION_GOOD = float(config['TRIMMING']['MIN_DURATION_GOOD'])
    PLOT_INF_PADDING = float(config['PLOTTING']['PLOT_INF_PADDING'])
    PLOT_SUP_PADDING = float(config['PLOTTING']['PLOT_SUP_PADDING'])
    VISUALLY_CONFIRM = config['PLOTTING']['VISUALLY_CONFIRM'].lower() == 'true'
    PLOT_EEG_SCALING = float(config['PLOTTING']['PLOT_EEG_SCALING'])
    DISCARD_FILE = eval(config['TRIMMING']['DISCARD_FILE'])
    ONLY_PAIRED_ANNOT = eval(config['TRIMMING']['ONLY_PAIRED_ANNOT'])


def _print_off():
    sys.stdout = open(devnull, 'w')


def _print_on():
    sys.stdout = sys.__stdout__


def get_file_paths(root: str) -> dict[str, str]:
    """
    Get all the EDF files under the root path. It searches recursively inside all the subdirectories.
    Each EDF file should have a unique subject identification in its name.
    :param root: Root path under which all the EDF files to process are located.
    :return: A dictionary with the EDF files paths as values and the subject identification as keys.
    """

    if not isdir(root):
        raise NotADirectoryError(f"There's no '{root}' directory. Please, check your config.ini file.")

    #all_files = glob.glob(f'{root}/**/*.edf', recursive=True)
    all_files = glob.glob(f'{root}/*', recursive=False)
    file_paths = {}
    for filepath in all_files:
        #subject_code = filepath.split(os_sep)[-1].split('.edf')[0]  # get file name without extension
        subject_code = filepath.split(os_sep)[-1]  # get subdirectory name
        file_paths[subject_code] = join(filepath, 'EEG/preprocessed/mat_format/RestingState.mat')
    return file_paths


def read_edf(filepath: str) -> mne.io.Raw:
    """
    Read an EDF file and return a Raw object.
    :param filepath: Path to the EDF file.
    :return: A mne.Raw object with raw data and metadata.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            return mne.io.read_raw_edf(filepath, infer_types=True, preload=False, verbose=False)
        except Exception as e:
            _print_on()
            print(f"Could not read file '{filepath}'.")
            _print_off()
            raise e


electrode_mapping = {
    36: 'C3',
    104: 'C4',
    'Cz': 'Cz',
    24: 'F3',
    124: 'F4',
    33: 'F7',
    122: 'F8',
    22: 'Fp1',
    9: 'Fp2',
    15: 'Fpz',
    11: 'Fz',
    70: 'O1',
    83: 'O2',
    52: 'P3',
    92: 'P4',
    62: 'Pz',
    58: 'T5',
    96: 'T6',
    45: 'T3',
    108: 'T4',
}

def read_mat(filepath: str) -> mne.io.Raw:
    """
    Read an MAT file and return a Raw object.
    :param filepath: Path to the MAT file.
    :return: A mne.Raw object with raw data and metadata.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            # Read MAT object with scipy
            mat = sio.loadmat(filepath)
            data = mat['result']['data'][0][0]  # channels x samples
            sf = float(mat['result']['srate'][0][0][0][0])
            channel_names = [x[0] for x in mat['result']['chanlocs'][0][0][0]['labels']]

            # Choose only the electrodes we want
            indexes_to_keep = [i for i, name in enumerate(channel_names) if name in electrode_mapping.keys() or int(name[1:]) in electrode_mapping.keys()]
            data = data[indexes_to_keep]
            channel_names = [name for i, name in enumerate(channel_names) if i in indexes_to_keep]
            channel_names_10_10 = [electrode_mapping[int(name[1:]) if 'E' in name else name] for name in channel_names]

            # Get events
            events = mat['result']['event'][0][0][0]
            events = [(e['type'][0], int(e['sample'][0]), 0) for e in events]
            annotations = mne.Annotations(onset=[e[1] / sf for e in events], duration=[e[2] for e in events], description=[e[0] for e in events])

            # Create info object
            info = mne.create_info(channel_names_10_10, sf, ch_types='eeg')
            raw = mne.io.RawArray(data, info)
            raw.set_annotations(annotations)
            raw = raw.resample(128)
            return raw

        except Exception as e:
            _print_on()
            print(f"Could not read file '{filepath}'.")
            print(e)
            _print_off()
            raise e


def plot(raw: mne.io.Raw, start: float, duration: float, title: str = ""):
    fig = raw.plot(start=start if start > 0 else 0, duration=duration, block=True, title=title,
                 verbose=False, scalings=PLOT_EEG_SCALING, order=relevant_channels, n_channels=len(relevant_channels),
                   highpass=1., lowpass=60.)
    return fig

def manual_decision(raw, annotation):
    _print_on()
    print(f"-----\nUNKNOWN ANNOTATION: '{annotation['description']}' at {annotation['onset']:.1f}s")
    print("Please, inspect this annotation in the plot that just opened. When ready to decide what to do, try to close it (although it might not) and answer the questions that will appear here in the Terminal.")
    _print_off()
    fig = plot(raw,
               start=annotation['onset'] - raw.first_time - PLOT_INF_PADDING,
               duration=PLOT_INF_PADDING + annotation['duration'] + PLOT_SUP_PADDING,
               title=f"What do we do with annotation '{annotation['description']}'?")

    def __answer_questions():
        try:
            # Exclude?
            to_exclude = '.'
            while to_exclude.lower() not in ('y', 'n'):
                to_exclude = input("To exclude? [y/n] ")
            to_exclude = to_exclude.lower() == 'y'

            correctness = '.'
            while True:
                # Details. Might raise ValueError if input types are not given correct
                if to_exclude:
                    inf_padding = float(input("Inferior padding: "))
                    sup_padding = float(input("Superior padding: "))
                    default_duration = float(input("Default duration: "))
                print("For next time...")
                sub_string = input("Look for a sub-string of this? [y/n/...] ")
                category = input("Any category?: ")

                # Confirm entries
                while correctness.lower() not in ('y', 'n'):
                    correctness = input("Is the above information correct? [y/n] ")
                correctness = correctness.lower() == 'y'

                # Return match
                if correctness:
                    if to_exclude:
                        return {'exclude': to_exclude, 'category': category, 'sub_string': sub_string,
                                'inf_padding': inf_padding, 'sup_padding': sup_padding,
                                'default_duration': default_duration}
                    else:
                        return {'exclude': to_exclude, 'category': category, 'sub_string': sub_string}
                else:
                    print("-- Ok, let's repeat.")
        except ValueError:
            print("-- Invalid input. Let's repeat.")
            return __answer_questions()
        except:
            print("-- Something went wrong. Let's repeat.")
            return __answer_questions()

    _print_on()
    res = __answer_questions()
    print(f"-----")
    _print_off()
    return res

def _process_label(label: str) -> str:
    label = label.replace(" ", "").lower()
    return label


def _get_pair_annotations_labels(annotations: dict) -> tuple[str]:
    """
    Returns all the labels that can be recognised as part of a pair.
    """
    labels = []
    for _, pair in annotations.items():
        for label in pair['start']:
            labels.append(label)
        for label in pair['end']:
            labels.append(label)
    return tuple(labels)


def get_annotation_pairs_intervals_to_discard(raw: mne.io.Raw, expected_annotations) -> tuple[list, dict, dict]:
    """
    Looks for annotation pairs as defined in PAIR_ANNOTATIONS_PATH file, and returns the intervals
    between the two annotations of each pair. Also returns the statistics of these intervals.
    Note: It deletes the found pairs from the annotations.
    """
    intervals_to_discard = []  # User can configure to discard the periods of non-interest
    intervals_to_separate = {}  # User can configure to keep these periods (now of interest) in a separate folder
    statistics_excluded = {}

    # Get all file annotations
    all_annotations = raw.annotations

    # Find all "starts" and "ends" annotations, group by category
    starts, ends = {}, {}
    for i in range(len(all_annotations)):
        annotation = all_annotations[i]
        label = _process_label(annotation['description'])
        for category, expected_annotation in expected_annotations.items():
            if any(_process_label(l) in label for l in expected_annotation['start']):
                if category not in starts:
                    starts[category] = [i, ]
                else:
                    starts[category].append(i)
            if any(_process_label(l) in label for l in expected_annotation['end']):
                if category not in ends:
                    ends[category] = [i, ]
                else:
                    ends[category].append(i)

    annotations_to_delete = []

    # Make pairs
    # We need one end for each start. This is verified by the onsets.
    # The default duration will be considered for the missing pairs.
    for category in starts.keys():
        category_starts = sorted(starts[category], key=lambda x: all_annotations[x]['onset'])
        if category in ends:
            category_ends = sorted(ends[category], key=lambda x: all_annotations[x]['onset'])
        else:
            category_ends = []

        # Match them by proximity in time
        for start_annotation_ix in category_starts:

            # Are there any ends after this start?
            candidate_ends = [a for a in category_ends if all_annotations[a]['onset'] - all_annotations[start_annotation_ix]['onset'] > 0]

            if len(candidate_ends) > 0:  # If there are ends of the same category after this start...
                # Find the closest
                closest_end_ix = min(category_ends, key=lambda x: all_annotations[x]['onset'] - all_annotations[start_annotation_ix]['onset'])
                # Mark for discard
                inf_boundary = all_annotations[start_annotation_ix]['onset'] - expected_annotations[category]['inf_padding']
                sup_boundary = all_annotations[closest_end_ix]['onset'] + expected_annotations[category]['sup_padding']
                # make sure the interval is within the file domain
                inf_boundary = inf_boundary if inf_boundary > raw.first_time else raw.first_time
                sup_boundary = sup_boundary if sup_boundary < raw.times[-1] else raw.times[-1]
                duration = sup_boundary - inf_boundary
                if expected_annotations[category]['exclude']:
                    intervals_to_discard.append((inf_boundary, sup_boundary))
                    #print(f"* Discarded {duration:.1f}s of '{category}'.")
                if expected_annotations[category]['separate_segment']:
                    if category not in intervals_to_separate:
                        intervals_to_separate[category] = [(inf_boundary, sup_boundary)]
                    else:
                        intervals_to_separate[category].append((inf_boundary, sup_boundary))
                    #print(f"* Will separate '{category}' segment in a sub-directory.")
                # Mark annotations for deletion
                annotations_to_delete += [start_annotation_ix, closest_end_ix]
                # Remove end from list (to ease the next search and to check if there are unpaired ends at the end)
                category_ends.remove(closest_end_ix)

            else:  # If there is not...
                # Use the default duration
                estimate_end = all_annotations[start_annotation_ix]['onset'] + expected_annotations[category]['default_duration']
                # Mark for discard
                inf_boundary = all_annotations[start_annotation_ix]['onset'] - expected_annotations[category]['inf_padding']
                sup_boundary = estimate_end + expected_annotations[category]['sup_padding']
                # make sure the interval is within the file domain
                inf_boundary = inf_boundary if inf_boundary > raw.first_time else raw.first_time
                sup_boundary = sup_boundary if sup_boundary < raw.times[-1] else raw.times[-1]
                duration = sup_boundary - inf_boundary
                if expected_annotations[category]['exclude']:
                    intervals_to_discard.append((inf_boundary, sup_boundary))
                    #print(f"* Discarded {duration:.1f}s of '{category}'.")
                if expected_annotations[category]['separate_segment']:
                    if category not in intervals_to_separate:
                        intervals_to_separate[category] = [(inf_boundary, sup_boundary)]
                    else:
                        intervals_to_separate[category].append((inf_boundary, sup_boundary))
                    #print(f"* Will separate '{category}' segment in a sub-directory.")
                # Mark start annotation for deletion
                annotations_to_delete.append(start_annotation_ix)

            # Count statistics
            if category not in statistics_excluded:
                statistics_excluded[category] = duration
            else:
                statistics_excluded[category] += duration

        # If there are any ends left, mark them for discard as well, using the default duration
        for end_annotation_ix in category_ends:
            estimate_start = all_annotations[end_annotation_ix]['onset'] - expected_annotations[category]['default_duration']
            # Mark for discard
            inf_boundary = estimate_start - expected_annotations[category]['inf_padding']
            sup_boundary = all_annotations[end_annotation_ix]['onset'] + expected_annotations[category]['sup_padding']
            # make sure the interval is within the file domain
            inf_boundary = inf_boundary if inf_boundary > raw.first_time else raw.first_time
            sup_boundary = sup_boundary if sup_boundary < raw.times[-1] else raw.times[-1]
            duration = sup_boundary - inf_boundary
            if expected_annotations[category]['exclude']:
                intervals_to_discard.append((inf_boundary, sup_boundary))
                #print(f"* Discarded {duration:.1f}s of '{category}'.")
            if expected_annotations[category]['separate_segment']:
                if category not in intervals_to_separate:
                    intervals_to_separate[category] = [(inf_boundary, sup_boundary)]
                else:
                    intervals_to_separate[category].append((inf_boundary, sup_boundary))
                #print(f"* Will separate '{category}' segment in a sub-directory.")
            # Mark end annotation for deletion
            annotations_to_delete.append(end_annotation_ix)
            # Count statistics
            if category not in statistics_excluded:
                statistics_excluded[category] = duration
            else:
                statistics_excluded[category] += duration

    # Delete annotations
    # raw.annotations.delete(annotations_to_delete)

    return intervals_to_discard, intervals_to_separate, statistics_excluded


def get_annotation_intervals_to_discard(raw: mne.io.Raw, expected_annotations: dict, labels_to_ignore: tuple) -> tuple[list, dict, dict]:
    """
    Discards irrelevant periods as given by the annotations.
    What to do with the period of each annotation is defined in the annotations.json file.
    For each item in the annotations.json file, if its label is case-insensitive found and "exclude" is true,
    then the period of the annotation is discarded, otherwise it is kept. The period of annotation is also padded by
    the "inf_padding" and "sup_padding" values.
    :param raw: Raw object to process.
    :param expected_annotations: List of annotations to expect.
    :return: Raw object with irrelevant periods discarded.
    """
    intervals_to_discard = []
    statistics_excluded = {}

    all_annotations = raw.annotations

    for annotation in all_annotations:

        # Ignore?
        if annotation['description'] == INTERRUPTION_LABEL:
            continue

        # Ignore?
        if any(_process_label(l) in _process_label(annotation['description']) for l in labels_to_ignore):
            continue

        # 1. Look for the annotation in the expected annotations list
        match = None  # no decision yet
        for expected_label, expected_annotation in expected_annotations.items():
            if expected_annotation['sub_string'] and _process_label(expected_label) in _process_label(annotation['description']):  # substring is enough
                match = expected_annotation
                break
            elif not expected_annotation['sub_string'] and _process_label(expected_label) == _process_label(annotation['description']):  # exact match
                match = expected_annotation
                break
        if match is None:  # This annotation is not in the expected list; IGNORE (NEW)
            print(f"* Did not know what to do with '{annotation['description']}'.")
            continue

        if match['exclude']:
            # 2. Mark for discard
            duration = annotation['duration'] if annotation['duration'] > 0 else match[
                'default_duration']  # use default duration if none was annotated
            inf_boundary = annotation['onset'] - match['inf_padding']
            sup_boundary = annotation['onset'] + duration + match['sup_padding']
            seconds_discarded = sup_boundary - inf_boundary
            intervals_to_discard.append((inf_boundary if inf_boundary > raw.first_time else raw.first_time, sup_boundary if sup_boundary < raw.times[-1] else raw.times[-1]))  # make sure the interval is within the file domain

            # 3. Count statistics
            if match['category'] not in statistics_excluded:
                statistics_excluded[match['category']] = seconds_discarded
            else:
                statistics_excluded[match['category']] += seconds_discarded

            # 4. Log:
            _print_on()
            #print(f"* Discarded {seconds_discarded:.1f}s around '{annotation['description']}'.")
            _print_off()

        else:
            _print_on()
            #print(f"* Ignored '{annotation['description']}'.")
            _print_off()

    return intervals_to_discard, expected_annotations, statistics_excluded


def check_if_discard_file(raw: mne.io.Raw, discard_annoations: list) -> Union[bool, str]:
    """
    Checks if the file should be discarded, based on the annotations given.
    """
    all_annotations = raw.annotations
    for annotation in all_annotations:
        if any(_process_label(l) in _process_label(annotation['description']) for l in discard_annoations):
            return str(annotation['description'])
    return False


def get_edge_intervals_to_discard(raw: mne.io.Raw) -> tuple[list, dict]:
    """
    Returns the edges of the files, as defined by the DISCARD_BEGIN and DISCARD_END variables.
    Also returns the statistics of these intervals.
    """
    return [(raw.first_time, raw.first_time + DISCARD_BEGIN), (raw.times[-1] - DISCARD_END, raw.times[-1])], {'Edges': DISCARD_BEGIN + DISCARD_END}


def _union_intervals(intervals):
    """
    Given a list of intervals, returns a list of intervals where all the overlapping intervals are joined.
    """
    intervals.sort(key=lambda x: x[0])
    union = []
    for interval in intervals:
        if len(union) == 0 or interval[0] > union[-1][1]:  # if the interval does not overlap with the last interval in the union
            union.append(interval)
        else:  # if the interval overlaps with the last interval in the union
            union[-1] = (union[-1][0], max(interval[1], union[-1][1]))
    return union


def _intervals_complement(intervals: list[tuple[float, float]], begin: float, end: float) -> list[tuple[float, float]]:
    """
    Given a list of intervals and the domain begin and end, returns the complement of those intervals.
    """
    intervals_complement = []

    if not intervals:  # Check if intervals list is empty
        return intervals_complement

    if intervals[0][0] > begin:  # if the first interval does not start at the beginning of the domain
        intervals_complement.append((begin, intervals[0][0]))
    for i in range(len(intervals) - 1):
        intervals_complement.append((intervals[i][1], intervals[i + 1][0]))
    if intervals[-1][1] < end:  # if the last interval does not end at the end of the domain
        intervals_complement.append((intervals[-1][1], end))
    return intervals_complement


def make_report_discarded_intervals(raw: mne.io.Raw, intervals_to_discard: list[tuple[float, float]]) -> mne.Report:
    """
    Creates a report with the intervals that were discarded.
    """
    to_discard = Annotations(onset=[interval[0] for interval in intervals_to_discard],
                                duration=[interval[1] - interval[0] for interval in intervals_to_discard],
                                description=['BAD'] * len(intervals_to_discard))
    discarded_segments = raw.crop_by_annotations(to_discard, verbose=False)
    # Create report
    _print_off()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        report = mne.Report(verbose=False)
        for seg in discarded_segments:
            seg.pick(eeg_channels)
            report.add_raw(seg, title=f"Discarded segment {seg.first_time:.1f} - {seg.times[-1]:.1f} s", psd=True, scalings=PLOT_EEG_SCALING, butterfly=True)
    _print_on()
    return report


def discard_intervals(raw: mne.io.Raw, intervals_to_keep:list[tuple[float, float]], cut_by_interruptions=True, visually_confirm=False) -> tuple[list[mne.io.Raw], list[tuple[float, float]], Union[list[str], None]]:
    """
    Receives a list of intervals to discard and returns a list of Raw objects, with these intervals discarded.
    If cut_by_interruptions is True, it will also cut the resulting segments by interruptions.
    If visually_confirm is True, it will plot the segments to discard and ask for confirmation before proceeding.
    In this plot, the user can check if the segments to discard are correct and adjust them as needed.
    """

    # Create 'GOOD' annotations for the intervals to keep
    to_keep = Annotations(onset=[interval[0] + DELTA for interval in intervals_to_keep],
                          duration=[interval[1] - interval[0] + DELTA for interval in intervals_to_keep],
                          description=['GOOD'] * len(intervals_to_keep),
                          orig_time=raw.annotations.orig_time)

    if len(intervals_to_keep) == 0:  # in the rare cases no good intervals to keep were found, the user is asked to visually confirm. In this way, they can mark segments to keep by visual inspection.
        _no_original_good_segments = True
        #visually_confirm = True
    else:
        _no_original_good_segments = False

    # If the user wants to confirm visually (optional)
    to_keep_adjusted = None
    adjustments = None
    if visually_confirm:
        _print_on()
        #print("In the plot that just opened, check the segments that are going to be kept (highlighted as 'GOOD').")
        #print("Upon visual inspection, press 'A', so you can make adjustments of 'GOOD' segments (drag their edges), or even delete some of them (right-click on them).")
        #print("To proceed, close it (although it might not).")
        _print_off()
        raw.set_annotations(raw.annotations + to_keep )
        if not _no_original_good_segments:
            plot(raw, start=to_keep[0]['onset']-10, duration=15, title="Segments to keep are highlighted as GOOD. Adjust if needed.")
        else:
            plot(raw, start=0, duration=15,
                 title="No segments to keep were found. Add some using the 'GOOD' label.")
        # get adjusted GOOD annotations
        adjusted_goods = [annotation for annotation in raw.annotations if annotation['description'] == 'GOOD']
        # Make annotations object out of them
        to_keep_adjusted = Annotations(onset=[annotation['onset'] for annotation in adjusted_goods],
                                        duration=[annotation['duration'] for annotation in adjusted_goods],
                                        description=['GOOD'] * len(adjusted_goods))

        # Keep track of what changed
        _print_on()
        print("If any, the adjustments made interactively were taken into account.")
        _print_off()
        good_intervals_before = [(f"\n[{before_good['onset']:.1f}, {(before_good['onset']+before_good['duration']):.1f}[ s") for before_good in to_keep]
        good_intervals_after = [(f"\n[{after_good['onset']:.1f}, {(after_good['onset']+after_good['duration']):.1f}[ s") for after_good in to_keep_adjusted]
        adjustments = ["'GOOD' segments before:", *good_intervals_before, "\n'GOOD' segments after:", *good_intervals_after]

    if to_keep_adjusted is None:
        to_keep_adjusted = to_keep

    # Add interruptions to the cutting (optional)
    if cut_by_interruptions:  # Split "to_keep_adjusted" annotations by interruptions
        _print_on()
        print("Cutting segments by interruptions...")
        _print_off()
        to_keep_new = Annotations(onset=[], duration=[], description=[], orig_time=raw.annotations.orig_time)
        # 1. Get all interruptions
        all_interruptions = [annotation for annotation in raw.annotations if annotation['description'] == INTERRUPTION_LABEL]
        all_interruptions = np.array(sorted([interr['onset'] for interr in all_interruptions]))
        # 2. Traverse each 'GOOD' annotation
        for a_ix, good_ann in enumerate(to_keep_adjusted):
            onset, offset = good_ann['onset'], good_ann['onset'] + good_ann['duration']
            # 3. Find interruptions within this annotation
            inside_interruptions = all_interruptions[(onset < all_interruptions) & (all_interruptions< offset)]
            if len(inside_interruptions) > 0:
                # 4. Pad 'inside_interruptions' with the edges of the annotation
                inside_interruptions = np.concatenate(([onset], inside_interruptions, [offset]))
                # 5. Create new smaller annotations, by clipping the original 'GOOD' annotation by the interruptions
                new_annotations = {'onset': [], 'duration': [], 'description': []}
                for i in range(len(inside_interruptions) - 1):
                    new_annotations['onset'].append(inside_interruptions[i])
                    new_annotations['duration'].append(inside_interruptions[i + 1] - inside_interruptions[i])
                    new_annotations['description'].append('GOOD')
                # 7. Add new annotations
                to_keep_new = to_keep_new + Annotations(onset=new_annotations['onset'],
                                                duration=new_annotations['duration'],
                                                description=new_annotations['description'],
                                                orig_time=raw.annotations.orig_time)
            else:
                # 4. Add the original 'GOOD' annotation
                to_keep_new = to_keep_new + Annotations(onset=[onset],
                                                duration=[offset - onset],
                                                description=['GOOD'],
                                                orig_time=raw.annotations.orig_time)

        # 8. Update 'to_keep' annotations
        to_keep_adjusted = to_keep_new

        # 9. reshape annotations onset and offset in order not to be larger than tmax (length of raw data)
        # Check and adjust duration if necessary
        if to_keep_adjusted:
            max_time = raw.times[-1]  # Get the maximum time from raw data
            if to_keep_adjusted.onset[-1] + to_keep_adjusted.duration[-1] > max_time:
                    to_keep_adjusted.duration[-1] = max_time - to_keep_adjusted.onset[-1]
            if to_keep_adjusted.onset[-1] >= max_time:
                    to_keep_adjusted.delete(len(to_keep_adjusted)-1)

        # 9.
        if len(to_keep_adjusted) > 0:
            max_time = raw.times[-1]
            if to_keep_adjusted.onset[-1] + to_keep_adjusted.duration[-1] > max_time:
                to_keep_adjusted.duration[-1] = max_time - to_keep_adjusted.onset[-1]

    # Crop the raw object by the 'GOOD' annotations
    _print_on()
    print("Trimming and creating smaller segments...")
    _print_off()
    segments = raw.crop_by_annotations(to_keep_adjusted, verbose=False)

    # Make list of intervals kept
    intervals_kept = [(good['onset'], good['onset'] + good['duration']) for good in to_keep_adjusted]

    return segments, intervals_kept, adjustments


def serialise_processed_files(processed_subjects=None):
    # Serialise processed subjects
    if processed_subjects:
        with open(PROCESSED_PATH, 'w') as file:
            file.write('\n'.join(processed_subjects))


def write_seconds_by_category_discarded(file, statistics_excluded):
    file.write("Seconds discarded by category:\n")
    for category, seconds in statistics_excluded.items():
        file.write(f"{category}: {seconds:.3f}\n")


def write_useful_seconds(file, former_seconds, discarded_seconds):
    useful_seconds = former_seconds - discarded_seconds
    file.write(f'{former_seconds:.1f} -> {useful_seconds:.1f} ({((discarded_seconds / former_seconds) * 100):.1f}% discarded)\n')


def write_file_discarded(file, annotation):
    file.write(f"100% discarded due to annotation '{annotation}'")


# ===================================
# Signal Quality Indexes

def eeg_quality_index(raw_matrix, window_length: int,
                      threshold_oha,
                      threshold_thv,
                      threshold_chv, rejection_cutoff_chv, rejection_ratio_chv):
    """
    Note: Normalise the signal between 0 and 1 first.
    Output: List of (oha, thv, chv) tuples ordered by windows.
    """

    def __chv(eeg_array, channel_threshold=1.5, rejection_cutoff=None, rejection_ratio=None):

        # 1. Remove timepoints of very high variance from channels
        if rejection_cutoff is not None:
            ignoreMask = np.logical_or(eeg_array > rejection_cutoff, eeg_array < -rejection_cutoff)
            onesPerChan = np.sum(ignoreMask, axis=1)
            onesPerChan = onesPerChan / eeg_array.shape[1]
            overRejRatio = onesPerChan > rejection_ratio
            ignoreMask[overRejRatio, :] = False
            eeg_array[ignoreMask] = np.nan

        # 2. Calculate CHV
        return np.sum(np.nanstd(eeg_array, axis=1) > channel_threshold) / eeg_array.shape[0]

    res = []
    windows_starts = range(0, raw_matrix.shape[1], window_length)
    for i in windows_starts:  # interate matrix by windows
        eeg_data = raw_matrix[:, i:i+window_length]
        # Evaluates the ratio of data points that exceed the absolute value a certain voltage amplitude.
        oha = np.sum(np.abs(eeg_data) > threshold_oha) / (eeg_data.shape[0] * eeg_data.shape[1])
        # Evaluates the ratio of time points in which the % of standard deviation across all channels exceeds a threshold.
        thv = np.sum(np.greater(np.std(eeg_data, axis=0, ddof=1), threshold_thv)) / eeg_data.shape[1]
        # Evaluates the ratio of channels in which the % of standard deviation across all timepoints exceeds a threshold.
        chv = __chv(eeg_data, threshold_chv, rejection_cutoff_chv, rejection_ratio_chv)
        # Append results
        res.append((oha, thv, chv))
    return res


def find_good_quality_intervals(raw, window_length: int, threshold_oha, threshold_thv, threshold_chv, rejection_cutoff, rejection_ratio) -> list[tuple[float, float]]:
    # Convert raw data to matrix
    raw_matrix = raw.get_data()  # output shape (n_channels, n_times)
    # Remove ECG and A1, A2 channels
    raw_matrix = raw_matrix[eeg_channels, :]
    # Normalise each channel between 0 and 1
    raw_matrix = (raw_matrix - raw_matrix.min(axis=1, keepdims=True)) / (raw_matrix.max(axis=1, keepdims=True) - raw_matrix.min(axis=1, keepdims=True))

    # Compute SQIs by window
    windows_starts = range(0, raw_matrix.shape[1], window_length)
    sqi_results_by_window = eeg_quality_index(raw_matrix, window_length, threshold_oha, threshold_thv, threshold_chv, rejection_cutoff, rejection_ratio)

    # Find good quality windows
    # Rule: (oha_sqi < 0.1) & (thv_sqi < 0.1) & (chv_sqi < 0.15)
    good_quality_windows = []
    for i, sqi in enumerate(sqi_results_by_window):
        #if (sqi[0] < 0.1) and (sqi[1] < 0.1) and (sqi[2] < 0.15):
        if (sqi[0] < 0.1) and (sqi[1] < 0.02) and (sqi[2] < 0.11):
            good_quality_windows.append(i)

    # Prepare a list of intervals, with initial reference at the first sample (tmin=0)
    intervals = []
    for i in good_quality_windows:
        intervals.append((i*window_length, (i+1)*window_length))

    # Combine adjacent intervals
    intervals = _union_intervals(intervals)

    return intervals


def _intersect_sets_of_intervals(intervals1, intervals2):
    intervals1.sort(key=lambda x: x[0])  # sort by start T
    intervals2.sort(key=lambda x: x[0])  # sort by start T

    intersection = []
    i, j = 0, 0
    while i < len(intervals1) and j < len(intervals2):
        if intervals1[i][1] <= intervals2[j][0]:
            i += 1
        elif intervals2[j][1] <= intervals1[i][0]:
            j += 1
        else:
            start = max(intervals1[i][0], intervals2[j][0])
            end = min(intervals1[i][1], intervals2[j][1])
            intersection.append((start, end))
            if intervals1[i][1] <= intervals2[j][1]:
                i += 1
            else:
                j += 1

    return intersection


def select_greedy(good, I):
    """
    Selects the 'I' longest intervals from the list of good intervals.
    :param good: List of good intervals.
    :param I: Number of intervals to select.
    :return: List of selected intervals.
    """
    good.sort(key=lambda x: x[1] - x[0], reverse=True)  # sort by duration
    return good[:I]


if __name__ == '__main__':

    mne.set_config('MNE_BROWSER_BACKEND', 'qt')
    # Use interactive backend for matplotlib
    #if sys.platform == 'darwin':  # for MacOS
    #    matplotlib.use('TkAgg')
    #elif sys.platform == 'win32':  # for Windows
    #    matplotlib.use('QtAgg')

    # Read config file
    read_config_file()

    # Load known paired annotations
    expected_pair_annotations = json.load(open(PAIR_ANNOTATIONS_PATH, 'r'))
    expected_single_annotations = json.load(open(SINGLE_ANNOTATIONS_PATH, 'r'))

    # Get all the subjects that have already been processed
    if isfile(PROCESSED_PATH):
        with open(PROCESSED_PATH, 'r') as file:
            processed_subjects = file.read().split('\n')
    else:
        processed_subjects = []

    # Get all the EDF files under the root path
    all_file_paths = get_file_paths(COMMON_PATH)

    # Process each file
    for subject_code, filepath in all_file_paths.items():
        if subject_code in processed_subjects:  # skip if already processed
            continue

        # 1. Read file
        _print_on()
        print(f"#####\nFile {subject_code}")
        _print_off()
        try:
            #raw = read_edf(filepath)  # 1. Read EDF file
            raw = read_mat(filepath)  # 1. Read MAT file
        except Exception as e:
            continue

        # 2. Discard file?
        discard_or_not = check_if_discard_file(raw, DISCARD_FILE)
        if isinstance(discard_or_not, str):
            _print_on()
            print(f"File discarded because of annotation '{discard_or_not}'.")
            print("Creating statistics...")
            _print_off()
            subject_output_path = join(OUTPUT_PATH, subject_code)
            if not isdir(subject_output_path):
                mkdir(subject_output_path)
            with open(f'{subject_output_path}/discarded.txt', 'w') as file:
                write_file_discarded(file, discard_or_not)
            processed_subjects.append(subject_code)
            serialise_processed_files(processed_subjects)
            continue
        elif isinstance(discard_or_not, bool) and not discard_or_not:
            pass

        former_duration = raw.times[-1] - raw.first_time
        statistics_excluded = {}
        intervals_to_discard = []

        # 3. Trim edges?
        if TRIM_EDGES:
            _print_on()
            #print("Trimming edges...")
            _print_off()
            intervals_to_discard_1, statistics_excluded_1 = get_edge_intervals_to_discard(raw)
            statistics_excluded.update(statistics_excluded_1)  # Merge statistics
            intervals_to_discard += intervals_to_discard_1

        # 4. Find intervals to discard, based only on paired annotations: good_A
        _print_on()
        intervals_to_discard_1, _, _ = get_annotation_pairs_intervals_to_discard(
            raw, expected_pair_annotations)
        intervals_to_discard += intervals_to_discard_1

        if not ONLY_PAIRED_ANNOT:
            intervals_to_discard_1, _, _ = get_annotation_intervals_to_discard(
                raw, expected_single_annotations, _get_pair_annotations_labels(expected_pair_annotations))
            _print_off()
            intervals_to_discard += intervals_to_discard_1

        intervals_to_discard = _union_intervals(intervals_to_discard)  # Join overlapping intervals
        good_A = _intervals_complement(intervals_to_discard, raw.first_time, raw.times[-1])  # Get the complement of the intervals to discard

        # 5. Find poor quality intervals, seeing the full signal: good_B
        """
        _print_on()
        window_length = int(raw.info['sfreq'] * 3)  # 3 seconds
        good_B = find_good_quality_intervals(raw, window_length=window_length,
                                             threshold_oha=0.5, threshold_thv=0.14, threshold_chv=0.01,
                                             rejection_cutoff=None, rejection_ratio=None)
        good_B = [(x/raw.info['sfreq'], y/raw.info['sfreq']) for x, y in good_B]
        _print_off()
        """

        # 6. Intersect good_A and good_B
        #good = _intersect_sets_of_intervals(good_A, good_B)
        good = good_A

        # 7. Greediness: Suggest the 2 longest intervals
        suggestions = select_greedy(good, 3)

        # 8. Present suggestions and discard the remaining signal
        segments, intervals_kept, _ = discard_intervals(raw, suggestions, cut_by_interruptions=True, visually_confirm=VISUALLY_CONFIRM)

        # 9. Save segments
        subject_output_path = join(OUTPUT_PATH, subject_code)
        if not isdir(subject_output_path):
            mkdir(subject_output_path)
        _print_on()
        print(f"Exporting new ({len(segments)}) files...")
        _print_off()
        for (i, segment), interval in zip(enumerate(segments), intervals_kept):
            # Check size of segment > MIN_DURATION_GOOD
            min_duration = MIN_DURATION_GOOD if MIN_DURATION_GOOD > 0 else 1
            if interval[1] - interval[0] < min_duration:
                _print_on()
                print(f"Segment {i} ({interval[1] - interval[0]:.1f}s < {MIN_DURATION_GOOD}s) not saved because it's too short.")
                _print_off()
                continue
            if not ONLY_PAIRED_ANNOT:
                segment.annotations.delete(range(len(segment.annotations)))  # delete all annotations
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                try:
                    mne.export.export_raw(f'{subject_output_path}/seg{i+1}_{int(interval[0])}_{int(interval[1])}.edf', segment, verbose=False, overwrite=True)
                except ValueError as e:
                    _print_on()
                    print(f"Segment {i} ({interval[1] - interval[0]:.1f}s) not saved because of error: {e}.")
                    _print_off()
                    continue

        # 10. Mark subject as processed and update file
        processed_subjects.append(subject_code)
        serialise_processed_files(processed_subjects)

        print("Done.")

        #break
