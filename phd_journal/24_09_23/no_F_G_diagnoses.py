import numpy as np
from utils import get_diagnoses
from read import read_all_features


not_F_G_sessions = []

# 1) Get all sessions
all_available_sessions = np.loadtxt('/Volumes/MMIS-Saraiv/Datasets/KJPP/raw/all_recordings_codes.txt', dtype=str)
all_used_sessions = (read_all_features('KJPP', multiples=True).index.str.split('$').str[0]).to_list()

# 2) Read diagnoses of those that have age and gender
all_diagnoses = get_diagnoses(all_available_sessions)

# 3) Which sessions do not have F or G diagnoses?
for session in all_available_sessions:
    if session in all_diagnoses:
        diagnoses = all_diagnoses[session]
        for diagnosis in diagnoses:
            if (not diagnosis.startswith('F')) and (not diagnosis.startswith('G')):
                not_F_G_sessions.append(session)


# 4) Print number of sessions without F or G diagnoses
print("Sessions without F or G diagnoses:", len(not_F_G_sessions))

# 5) Get intersection of used sessions and not_F_G_sessions
intersection = list(set(all_used_sessions).intersection(set(not_F_G_sessions)))
print("Length of intersection:", len(intersection))

"""
# 6) Keep random 85% of the intersection set, discard the rest (use fixed random seed)
np.random.seed(42)
np.random.shuffle(intersection)
percent_85 = int(0.70 * len(intersection))
percent_15 = len(intersection) - percent_85
intersection_85 = intersection[:percent_85]

# 7) Keep random 15% of the not_F_G_sessions not in the intersection set, discard the rest (use fixed random seed)
np.random.seed(42)
np.random.shuffle(not_F_G_sessions)
not_F_G_sessions_wo_intersection = list(set(not_F_G_sessions).difference(set(intersection)))
not_F_G_sessions_wo_intersection_15 = not_F_G_sessions_wo_intersection[:percent_15]

# 8) Union of two sets
print("Length of intersection_85:", len(intersection_85))
print("Length of not_F_G_sessions_wo_intersection_15:", len(not_F_G_sessions_wo_intersection_15))
final_sessions = list(set(intersection_85).union(set(not_F_G_sessions_wo_intersection_15)))
print("Length of final_sessions:", len(final_sessions))
"""
final_sessions = intersection

# 9) Print final sessions
for session in final_sessions:
    print(session)
