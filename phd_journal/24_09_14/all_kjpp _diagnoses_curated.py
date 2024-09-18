from glob import glob
from os.path import exists

import pandas as pd
from pandas import DataFrame
from pandas import read_csv
import simple_icd_10 as icd


out_filepath = '/Volumes/MMIS-Saraiv/Datasets/KJPP/all_diagnoses_curated.csv'


diagnoses_groups = {
    # Mental Disorders (usually of adult onset)
    'Psychotic Disorders': ['F06.0', 'F06.2', 'F23', 'F24', 'F28', 'F29'],
    'Schizo and Delusional Disorders': ['F20', 'F21', 'F22', 'F25'],
    'Mood Disorders': ['F06.3', 'F30', 'F31', 'F34', 'F39'],
    'Depressive Disorders': ['F32', 'F33'],
    'Anxiety Disorders': ['F06.4', 'F40', 'F41'],
    'Obsessive-Compulsive Disorders': ['F42'],
    'Stress-related Disorders': ['F43'],
    'Dissociative Disorders': ['F44'],
    'Somatoform Disorders': ['F45'],
    'Cognitive Disorders': ['F06.7', ],
    'Personality and Behaviour Disorders': ['F07', 'F59', 'F60', 'F63'],
    'Mental and Behavioural from Psychoactives': ['F10', 'F11', 'F12', 'F13', 'F14', 'F15', 'F16', 'F17', 'F18', 'F19'],
    'Eating Disorders': ['F50'],
    'Mental Sleep Disorders': ['F51'],
    'Other Mental Disorders': ['F06.8', 'F09', 'F48', 'F54'],

    # Mental Disorders (usually of child onset)
    'Intellectual Disabilities': ['F70', 'F71', 'F72', 'F73', 'F78', 'F79'],
    'Developmental Speech and Language Disorders': ['F80', ],
    'Developmental Scholastic Disorders': ['F81', ],
    'Developmental Motor Disorders': ['F82', ],
    'Developmental Pervasive Disorders': ['F84', ],
    'Attention-deficit Hyperactivity Disorders': ['F90', ],
    'Conduct Disorders': ['F91', ],
    'Emotional Disorders': ['F93', ],
    'Tic disorders': ['F95', ],
    'Other Developmental Disorders': ['F88', 'F89', 'F98'],

    # Neurological Disorders
    'Epilepsies and Status Epilepticus': ['G40', 'G41'],
    'Migranes and Headaches': ['G43', 'G44'],
    'Ischemic and Vascular Brain Syndromes': ['G45', 'G46'],
    'Sleep Disorders': ['G47'],
    'CNS Inflammatory Diseases': ['G01', 'G02', 'G03', 'G04', 'G05', 'G06', 'G07', 'G08', 'G09'],
    'CNS Atrophies': ['G10', 'G11', 'G12', 'G13', 'G14'],
    'Extrapyramidal and Movement Disorders': ['G23', 'G24', 'G25', 'G26'],
    'CNS Demyelinating Diseases': ['G35', 'G36', 'G37', 'G38', 'G40'],
    'Neuropathies and Plexopathies': ['G50', 'G51', 'G52', 'G53', 'G54', 'G55', 'G56', 'G57', 'G58', 'G59', 'G60', 'G61', 'G62', 'G63', 'G64'],
    'Myo(neuro)pathies': ['G70', 'G71', 'G72', 'G73'],
    'Cerebral Palsy and Paralytic Syndromes': ['G80', 'G81', 'G82', 'G83'],
    'Other CNS or PNS Disorders': ['G90', 'G91', 'G92', 'G93', 'G94', 'G95', 'G96', 'G97', 'G98', 'G99'],

    # Other groups
    'Nutrition and Metabolic Disorders': ['E' + str(i).zfill(2) for i in range(40, 91)],
    'Endocrine Disorders': ['E' + str(i).zfill(2) for i in range(0, 40)],
    'Liver Diseases': ['K70', 'K71', 'K72', 'K73', 'K74', 'K75', 'K76', 'K77'],
    'Congenital Nervous Malformations': ['Q00', 'Q01', 'Q02', 'Q03', 'Q04', 'Q05', 'Q06', 'Q07'],
    'Chromosomal Abnormalities': ['Q90', 'Q91', 'Q92', 'Q93', 'Q95', 'Q96', 'Q97', 'Q98', 'Q99'],
}
N_DIAGNOSES_GROUPS = len(diagnoses_groups)


def get_group(query_code: str):
    if not icd.is_valid_item(query_code):
        return None
    else:
        for group, group_codes in diagnoses_groups.items():
            for code in group_codes:
                if query_code == code or icd.is_descendant(query_code, code):
                    return group
        print(f"Code {query_code} is valid but it was not found in any group.")
        return None


# Resume DataFrame
if exists(out_filepath):
    df = read_csv(out_filepath, sep=';')
    # Ensure the columns for diagnoses groups are present
    """
    if len(df.columns) != 7 + N_DIAGNOSES_GROUPS:
        for group in diagnoses_groups.keys():
            if group not in df.columns:
                # add it to the end, all with nans
                df[group] = None
                print(f"Diagnoses group '{group}' column was missing.")
    """
else:
    df = DataFrame(columns=['PATIENT', 'SESSION', 'GENDER', 'AGE', 'EMU', 'MEDICATION', 'ALL DIAGNOSES CODES'])
                           #+ list(diagnoses_groups.keys()))



# Read metadata_as_given
metadata = read_csv('/Volumes/MMIS-Saraiv/Datasets/KJPP/metadata_with_letters.csv', index_col=0, sep=',')

# Get list of all session codes
all_eeg_sessions = metadata.index.tolist()

if len(df) > 0:
    all_eeg_sessions = list(set(all_eeg_sessions) - set(df['SESSION']))

print("Number of files done:", len(df))
print("Number of files left:", len(all_eeg_sessions))


not_found = 0
no_report_read = 0
for eeg_session in all_eeg_sessions:
    print("############################################")
    print(eeg_session)

    # Find if it is in metadata
    if eeg_session not in metadata.index:
        print(f"Session {eeg_session} not found in metadata.")
        not_found += 1
        continue

    row = metadata.loc[eeg_session]
    patient_code = row['PATIENT']
    gender = row['GENDER']
    age = row['EEG AGE MONTHS'] / 12

    # Check in metadata if columns STUDY or UNIT contain KI.3 or KI3
    study = str(metadata.loc[eeg_session]['STUDY'])
    unit = str(metadata.loc[eeg_session]['UNIT'])
    emu = 'KI.3' in study or 'KI3' in study or 'KI.3' in unit or 'KI3' in unit

    # Check if any report was read
    if row.iloc[5:].isna().all():
        print(f"No report read for session {eeg_session}.")
        no_report_read += 1
        diagnoses = 'no report'
        medication = 'no report'
        df.loc[len(df)] = [patient_code, eeg_session, gender, age, emu, medication, diagnoses]  #+ [None] * N_DIAGNOSES_GROUPS
    else:
        diagnoses = []
        # only MAS1 to MAS4 (inclusive)
        for axis, d in row.iloc[5:20].items():
            if not pd.isna(d):
                # fit d to the expected format 'X00.0' or 'X00'
                d = d.replace(' ', '')
                d = d.replace('(', '')
                d = d.replace('\t', '')
                d = d.upper()
                if icd.is_valid_item(d):
                    diagnoses.append(d)
                    print(f"Valid >{d}<: {icd.get_description(d)}")
                else:
                    # Is it the German Modification?
                    # ICD-10-GM codes have a 2nd digit at the end, like 'X00.00', we'll remove it
                    if len(d.split('.')[-1]) > 1:
                        d_international = d + d.split('.')[-1][0]  # keep only first digit after the dot
                        if icd.is_valid_item(d_international):
                            diagnoses.append(d_international)
                            print(f"Valid >{d_international}< (originally >{d}<): {icd.get_description(d_international)}")
                            continue
                        else:
                            print("Tried GM 'X00.00' -> 'X00.0', but still not valid vvv")
                    # They can also have a subsection, 'X00.0', when the international code does not, 'X00'
                    d_international = d.split('.')[0]  # discard the subsection
                    if icd.is_valid_item(d_international):
                        diagnoses.append(d_international)
                        print(f"Valid >{d_international}< (originally >{d}<): {icd.get_description(d_international)}")
                        continue
                    else:
                        print("Tried GM 'X00.0'->'X00', but still not valid vvv")

                    print(f"{axis} >{d}<")
                    validated = False
                    while not validated:
                        answer = input("Enter the correct code ('u' if unreadable; 'n' if no code needed): ")
                        if answer == 'u':
                            d = f"{axis}|{d}"  # keep the same; maybe we'll know later
                            validated = True
                            diagnoses.append(d)
                        elif answer == 'n':
                            d = None
                            validated = True
                        else:
                            d = answer
                            if icd.is_valid_item(d):
                                validated = True
                                print(f"Valid >{d}<: {icd.get_description(d)}")
                                diagnoses.append(d)
                            else:
                                print(f"Invalid code >{d}<")

        medication = f"'{row['MEDICATION']}'"

        # Append row to DataFrame with first 7 columns (general columns)
        i = len(df)
        df.loc[i] = [patient_code, eeg_session, gender, age, emu, medication, diagnoses] # + [False] * N_DIAGNOSES_GROUPS

        """
        # Detail diagnoses: mark as True the group of the diagnosis (it can be more than one)
        for d in diagnoses:
            try:
                group = get_group(d)
                df.loc[i, group] = True
            except ValueError:
                print(f"No group found for >{d}<")
                continue
        """

    # save at every row
    df.to_csv(out_filepath, index=False, sep=';')

    print("Saved.")

print(f"Sessions not found in metadata: {not_found}")
print(f"Sessions without report read: {no_report_read}")
print(f"Sessions with report read: {len(df)}")

# Save
