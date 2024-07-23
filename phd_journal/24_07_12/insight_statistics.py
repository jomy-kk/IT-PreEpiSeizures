import pandas as pd

from read import read_all_features, read_mmse, read_ages

# 1) Read features
features = read_all_features('INSIGHT', multiples=False)
print("Features Shape:", features.shape)

# 2) Read targets
insight_targets = read_mmse('INSIGHT')
targets = pd.Series()
for index in features.index:
    if '$' in str(index):  # Multiples
        key = str(index).split('$')[0]  # remove the multiple
    else:  # Original
        key = index

    if '_' in str(key):  # insight
        key = int(key.split('_')[0])
        if key in insight_targets:
            targets.loc[index] = insight_targets[key]

targets = targets.dropna()  # Drop subject_sessions with nans targets
features = features.loc[targets.index]

# 3) Read ages
insight_ages = read_ages('INSIGHT')
ages = pd.Series()
for index in features.index:
    if '$' in str(index):  # Multiples
        key = str(index).split('$')[0]  # remove the multiple
    else:  # Original
        key = index

    if '_' in str(key):  # insight
        key = int(key.split('_')[0])
        if key in insight_ages:
            ages.loc[index] = insight_ages[key]

# 4) Concatenate all columns
data = pd.concat([features, targets, ages], axis=1)
# column names
data.columns = features.columns.tolist() + ['MMSE', 'Age']

# Indexes are "X_Y"
# We want to remove the "_Y" part and if there are multiple indexes with the same X, we want to keep the first one
data.index = [str(index).split('_')[0] for index in data.index]
data = data[~data.index.duplicated(keep='first')]
data.index = data.index.astype(int)


# 5) Statistics

# 5.1) Age (mean, std)
print("Age Statistics:")
print("Mean:", data['Age'].mean())
print("Std:", data['Age'].std())

# 5.2) MMSE (mean, std)
print("MMSE Statistics:")
print("Mean:", data['MMSE'].mean())
print("Std:", data['MMSE'].std())

# CSV with ages and MMSE
data[['Age', 'MMSE']].to_csv('insight_ages_mmse.csv')
