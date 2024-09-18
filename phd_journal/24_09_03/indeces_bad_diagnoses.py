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

F8 = []
F7 = []
F9 = []
F4 = []
EPILEPSIES = []
Q = []
G = []
K = []
E = []
no_report = []

for session in metadata.index:
    diagnoses = metadata.loc[session, 'ALL DIAGNOSES CODES']
    if '[' in diagnoses:
        diagnoses = eval(diagnoses)
        for diagnosis in diagnoses:
            if '|' in diagnosis:
                continue  # this is not a valid ICD-10 code
            if diagnosis.startswith('F8'):
                F8.append(session)
            if diagnosis.startswith('F7'):
                F7.append(session)
            if diagnosis.startswith('G40') or diagnosis.startswith('G47.4'):
                EPILEPSIES.append(session)
            if diagnosis.startswith('Q'):
                Q.append(session)
            if diagnosis.startswith('G') and not (diagnosis.startswith('G40') or diagnosis.startswith('G47.4')):
                G.append(session)
            if diagnosis.startswith('K'):
                K.append(session)
            if diagnosis.startswith('E'):
                E.append(session)

            if diagnosis.startswith('F9') and session not in F8 and session not in F7 and session not in EPILEPSIES and session not in Q:
                F9.append(session)

            if diagnosis.startswith('F4') and session not in F8 and session not in F7 and session not in EPILEPSIES and session not in Q:
                F4.append(session)

    elif diagnoses == 'no report':
        no_report.append(session)

F8 = np.array(list(set(F8)))
F7 = np.array(list(set(F7)))
F9 = np.array(list(set(F9)))
F4 = np.array(list(set(F4)))
EPILEPSIES = np.array(list(set(EPILEPSIES)))
Q = np.array(list(set(Q)))
G = np.array(list(set(G)))
K = np.array(list(set(K)))
E = np.array(list(set(E)))
no_report = np.array(list(set(no_report)))

# save
np.savetxt('F8.txt', F8, fmt='%s', delimiter='\n')
np.savetxt('F7.txt', F7, fmt='%s', delimiter='\n')
np.savetxt('F9.txt', F9, fmt='%s', delimiter='\n')
np.savetxt('F4.txt', F4, fmt='%s', delimiter='\n')
np.savetxt('EPILEPSIES.txt', EPILEPSIES, fmt='%s', delimiter='\n')
np.savetxt('Q.txt', Q, fmt='%s', delimiter='\n')
np.savetxt('G.txt', G, fmt='%s', delimiter='\n')
np.savetxt('K.txt', K, fmt='%s', delimiter='\n')
np.savetxt('E.txt', E, fmt='%s', delimiter='\n')
np.savetxt('no_report.txt', no_report, fmt='%s', delimiter='\n')
