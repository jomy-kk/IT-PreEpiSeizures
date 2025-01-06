import pandas as pd
import numpy as np


x = pd.read_csv("/Users/saraiva/Desktop/CombatManuscript-3.csv")
x = x.iloc[:1596]

Y = []
for n_datasets in (2, 3, 4, 5, 6):
    y = x[x['Datasets'].apply(lambda x: len(eval(x)) == n_datasets)]
    Y.append(y)

# Define number of datasets
n_datasets = 4
print("Number of datasets:", n_datasets)

combat = Y[n_datasets-2][Y[n_datasets-2]['Method'] == 'neuroharmonize']
none = Y[n_datasets-2][Y[n_datasets-2]['Method'] == 'none']

diffs = []
for n_pc in range(2, 16):
    _none = none[none['Features'].apply(lambda x: len(eval(x)) == n_pc)][['Datasets', 'F1-Score']]
    _combat = combat[combat['Features'].apply(lambda x: len(eval(x)) == n_pc)][['Datasets', 'F1-Score']]
    for i in range(len(_none)):
        print('Datasets:', eval(_none.iloc[i]['Datasets']))
        print('None:', _none.iloc[i]['F1-Score'])
        print('Combat:', _combat.iloc[i]['F1-Score'])

    diff = _combat['F1-Score'].mean() - _none['F1-Score'].mean()
    print("Number od PC:", n_pc)
    print("Increase:", diff, '\n')
    diffs.append(diff)
print("Average:", sum(diffs)/len(diffs))
# Sort diffs and args
diffs = np.array(diffs)
args = np.argsort(diffs)[::-1]
diffs = diffs[args]
print("1st Max:", diffs[0], "at", args[0]+2, 'PCs')
print("2nd Max:", diffs[1], "at", args[1]+2, 'PCs')
print("3rd Max:", diffs[2], "at", args[2]+2, 'PCs')