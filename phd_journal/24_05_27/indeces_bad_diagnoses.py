import numpy as np
import pandas as pd

metadata = pd.read_csv('/Volumes/MMIS-Saraiv/Datasets/KJPP/curated_metadata.csv', index_col=1, sep=';', header=0)

# there's a column called "ALL DIAGNOSES CODES", which contain a list or 'no report' string
# when it contains a list we want to check if they contain any bad diagnoses or maybe-bad diagnoses

# bad diagnoses codes
# F80, F81, F82, F83, F84, F88, F89
# F70, F71, F72, F73, F78, F79
# G40, G47.4
# all Q codes

# maybe-bad diagnoses codes
# all other G codes
# all K codes
# all E codes

bad_diagnoses = []
maybe_bad_diagnoses = []
no_report = []

for session in metadata.index:
    diagnoses = metadata.loc[session, 'ALL DIAGNOSES CODES']
    if '[' in diagnoses:
        diagnoses = eval(diagnoses)
        for diagnosis in diagnoses:
            if '|' in diagnosis:
                continue  # this is not a valid ICD-10 code
            if diagnosis.startswith('F8') or diagnosis.startswith('F7') or diagnosis.startswith('G40') or diagnosis.startswith('G47.4') or diagnosis.startswith('Q'):
                bad_diagnoses.append(session)
            elif diagnosis.startswith('G') or diagnosis.startswith('K') or diagnosis.startswith('E'):
                maybe_bad_diagnoses.append(session)
            else:
                continue
    elif diagnoses == 'no report':
        no_report.append(session)

bad_diagnoses = np.array(list(set(bad_diagnoses)))
maybe_bad_diagnoses = np.array(list(set(maybe_bad_diagnoses)))
no_report = np.array(list(set(no_report)))

# save
np.savetxt('/Volumes/MMIS-Saraiv/Datasets/KJPP/session_ids/bad_diagnoses.txt', bad_diagnoses, fmt='%s', delimiter='\n')
np.savetxt('/Volumes/MMIS-Saraiv/Datasets/KJPP/session_ids/maybe_bad_diagnoses.txt', maybe_bad_diagnoses, fmt='%s', delimiter='\n')
np.savetxt('/Volumes/MMIS-Saraiv/Datasets/KJPP/session_ids/no_report.txt', no_report, fmt='%s', delimiter='\n')
