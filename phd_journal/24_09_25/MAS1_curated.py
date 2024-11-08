from glob import glob
from os.path import exists

import pandas as pd
from pandas import DataFrame
from pandas import read_csv
import simple_icd_10 as icd


in_filepath = '/Users/saraiva/Desktop/Doktorand/KJPP/all_diagnoses_curated.csv'
out_filepath = '/Users/saraiva/Desktop/Doktorand/KJPP/all_diagnoses_curated_withMAS11.csv'

# Resume DataFrame
if exists(in_filepath):
    df = read_csv(in_filepath, sep=';')
else:
    df = DataFrame(columns=['PATIENT', 'SESSION', 'GENDER', 'AGE', 'EMU', 'MEDICATION', 'ALL DIAGNOSES CODES'])
                           #+ list(diagnoses_groups.keys()))

# Read metadata_as_given
metadata = read_csv('/Users/saraiva/Desktop/Doktorand/KJPP/metadata.csv', sep=';')

# Get list of all session codes
all_eeg_sessions = metadata.index.tolist()

not_found = 0
no_report_read = 0
for eeg_session in all_eeg_sessions:
    print("############################################")
    print(eeg_session)

    # raw
    row = metadata.loc[eeg_session]

    # already curated
    this_curated = df.loc[df['SESSION'] == eeg_session]
    this_curated_index = this_curated.index.tolist()[0]
    this_curated = this_curated.iloc[0]

    # Check if any report was read
    if this_curated["MEDICATION"] == 'no report':
        continue
    else:
        diagnoses = []
        # only MAS1
        axis = 'MAS1'
        for d in [row["MAS11"], ]:
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

        # Append MAS11 to the diagnoses already there
        i = len(df)
        this_curated['ALL DIAGNOSES CODES'] = diagnoses + eval(this_curated['ALL DIAGNOSES CODES'])
        df.loc[this_curated_index] = this_curated

    # save at every row
    df.to_csv(out_filepath, index=False, sep=';')

    print("Saved.")

print(f"Sessions not found in metadata: {not_found}")
print(f"Sessions without report read: {no_report_read}")
print(f"Sessions with report read: {len(df)}")
