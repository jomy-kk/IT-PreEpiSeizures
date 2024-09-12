import seaborn as sns
import simple_icd_10 as icd
from matplotlib import pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from read import *
from utils import diagnoses_groups


# make a colour for each diagnoses supergroup
diagnoses_supergroups = ((0, 15), (15, 25), (25, 37), (37, 41))
diagnoses_supergroups_colors = ['red', 'blue', 'green', 'orange']
no_diagnosis_color = 'gray'

def get_colors(subject_codes):

    # Statistics counters
    N_no_list, N_empty_list, N_not_found = 0, 0, 0
    not_found = []


    # Read diagnoses
    diagnoses = read_diagnoses('KJPP')
    colors = []
    for code in subject_codes:
        code = code.split('$')[0]
        subject_diagnoses = diagnoses[code]
        color_decided = no_diagnosis_color
        #print("Code", code)
        #print("Subject Diagnoses:", subject_diagnoses)
        #print("Subject Diagnoses len:", len(subject_diagnoses))
        if isinstance(subject_diagnoses, list):
            for d in subject_diagnoses:
                if icd.is_valid_item(d):
                    #print("d =", d)
                    for i, D in enumerate(diagnoses_groups.values()):
                        #print("D =", D)
                        for el_D in D:
                            if icd.is_valid_item(el_D) and icd.is_descendant(d, el_D): # diagnosis belongs to this group
                                for j in range(len(diagnoses_supergroups)):
                                    if diagnoses_supergroups[j][0] < i < diagnoses_supergroups[j][1]:
                                        color_decided = diagnoses_supergroups_colors[j]
                                        break
                                break

        colors.append(color_decided)

        if not isinstance(subject_diagnoses, list):
            N_no_list += 1
        elif len(subject_diagnoses) == 0:
            N_empty_list += 1
        elif color_decided == no_diagnosis_color:
            N_not_found +=1
            not_found.append(subject_diagnoses)

    print("Statistics")
    print("No list: ", N_no_list)
    print("Empty list: ", N_empty_list)
    print("Not found: ", N_not_found)
    print(not_found)

    return colors


out_path = './scheme1/cv'

# Get all sub-directories
all_folds = glob(join(out_path, '*'), recursive=False)
all_folds.sort()

for fold_path in all_folds:
    fold_number = fold_path.split('/')[-1]
    print(f"Fold {fold_number}")

    predictions_targets = pd.read_csv(join(fold_path, 'predictions_targets.csv'), index_col=0)
    predictions, targets = list(predictions_targets['predictions']), list(predictions_targets['targets'])
    session_codes = list(predictions_targets.index)

    # Test the model
    print(f"Test examples: {len(predictions)}")

    colors = get_colors(session_codes)

    #print(targets)
    #print(predictions)
    #print(colors)

    # Make regression plot of test set
    plt.figure(figsize=(6, 5))
    plt.rcParams['font.family'] = 'Arial'
    sns.regplot(x=targets, y=predictions, scatter_kws={'alpha': 0.3, 'color': colors},
                line_kws={'color': '#C60E4F'})
    plt.xlabel('True Age (years)', fontsize=12)
    plt.ylabel('Predicted Age (years)', fontsize=12)
    plt.xlim(2, 24)
    plt.ylim(2, 24)
    plt.xticks([3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23], fontsize=11)
    plt.yticks([3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23], fontsize=11)
    plt.grid(linestyle='--', alpha=0.4)
    plt.box(False)
    plt.tight_layout()
    plt.savefig(join(fold_path, 'test.png'))

    # Compute errors by color
    predictions_targets['color'] = colors
    for color in diagnoses_supergroups_colors + [no_diagnosis_color, ]:
        print(color)
        this_color_predictions_targets = predictions_targets['color' == color]
        # Metrics
        mae = mean_absolute_error(this_color_predictions_targets['targets'], this_color_predictions_targets['predictions'])
        mse = mean_squared_error(this_color_predictions_targets['targets'], this_color_predictions_targets['predictions'])
        r2 = r2_score(this_color_predictions_targets['targets'], this_color_predictions_targets['predictions'])
        print(mae)
        print(mse)
        print(r2)

