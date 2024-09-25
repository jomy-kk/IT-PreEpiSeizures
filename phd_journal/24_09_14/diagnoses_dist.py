import numpy as np
import pandas as pd
import simple_icd_10 as icd

from read import read_all_features, read_ages
from utils import get_diagnoses

# 1) Read features
#features = read_all_features('KJPP', multiples=True)
#features = features.dropna(axis=0)
#sessions = features.index

# 1) Get all sessions
sessions = np.loadtxt('/Users/saraiva/Desktop/Doktorand/KJPP/woFG_tochristoph_old.txt', dtype=str)

# 2) Read diagnoses of those that have age and gender
all_diagnoses = get_diagnoses(sessions)

# 3) Count diagnoses distribution
counts = {}
for session in all_diagnoses.keys():
    diagnoses = all_diagnoses[session]
    for diagnosis in diagnoses:
        if diagnosis not in counts:
            counts[diagnosis] = 0
        counts[diagnosis] += 1

# 4) Group diagnoses by category
categories = {}
for diagnosis in counts.keys():
    ancestors = icd.get_ancestors(diagnosis)
    try:
        category = ancestors[-2]
    except Exception:
        category = ancestors[0]
    category = icd.get_description(category)
    if category not in categories.keys():
        categories[category] = []
    categories[category].append(diagnosis)

# 4) Sort diagnoses in categories by frequency
categories_sorted = {}
for category in categories.keys():
    diagnoses = categories[category]
    diagnoses = sorted(diagnoses, key=lambda x: counts[x], reverse=True)
    categories_sorted[category] = diagnoses

# 5) Get textual description of diagnoses
textual_descriptions = {}
for diagnosis in counts.keys():
    textual_descriptions[diagnosis] = icd.get_description(diagnosis)

# 6) Print sum of counts
print("Total diagnoses:", sum(counts.values()))

# 7) Print diagnoses distribution by category, and in each category by frequency
for letter in range(65, 91):
    letter = chr(letter)
    print("# Chapter", letter)
    for category, category_diagnoses in categories_sorted.items():
        if category_diagnoses[0][0] == letter:
            sum_category = sum([counts[diagnosis] for diagnosis in category_diagnoses])
            print("##", category, sum_category)
            for diagnosis in category_diagnoses:
                if letter == 'F' or letter == 'G':
                    print('*', diagnosis, textual_descriptions[diagnosis], counts[diagnosis])
                pass
            print()

    print()



