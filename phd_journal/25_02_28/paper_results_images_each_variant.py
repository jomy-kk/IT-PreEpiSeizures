import itertools
import gc
import shutil

import neptune
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC

from paper_results_each_variant import make_results
from paper_images_each_variant import make_images

n_datasets = 6
dataset_combination = ['Izmir', 'Newcastle', 'Miltiadous', 'Istambul', 'BrainLat:CL', 'BrainLat:AR']
variant = 'neuroharmonize'
#variant = 'none'

model_family = 'MLP'
models = (
    MLPClassifier(hidden_layer_sizes=(100, ), alpha=0.0001, activation='relu', max_iter=1000),
    MLPClassifier(hidden_layer_sizes=(10, ), alpha=0.0001, activation='relu', max_iter=1000),
    MLPClassifier(hidden_layer_sizes=(50, ), alpha=0.0001, activation='relu', max_iter=1000),
    MLPClassifier(hidden_layer_sizes=(200, ), alpha=0.0001, activation='relu', max_iter=1000),
    MLPClassifier(hidden_layer_sizes=(50, 50), alpha=0.0001, activation='relu', max_iter=1000),
    MLPClassifier(hidden_layer_sizes=(10, 10), alpha=0.0001, activation='relu', max_iter=1000),
    MLPClassifier(hidden_layer_sizes=(10, 25), alpha=0.0001, activation='relu', max_iter=1000),
    MLPClassifier(hidden_layer_sizes=(25, 10), alpha=0.0001, activation='relu', max_iter=1000),
    MLPClassifier(hidden_layer_sizes=(25, 25), alpha=0.0001, activation='relu', max_iter=1000),
    MLPClassifier(hidden_layer_sizes=(25, 100, 25), alpha=0.0001, activation='relu', max_iter=1000),
    MLPClassifier(hidden_layer_sizes=(10, 50, 10), alpha=0.0001, activation='relu', max_iter=1000),
    MLPClassifier(hidden_layer_sizes=(30, 20, 10), alpha=0.0001, activation='relu', max_iter=1000),
    MLPClassifier(hidden_layer_sizes=(100, 100), alpha=0.0001, activation='relu', max_iter=1000),
    MLPClassifier(hidden_layer_sizes=(200, 100), alpha=0.0001, activation='relu', max_iter=1000),
    MLPClassifier(hidden_layer_sizes=(200, 100), alpha=0.0001, activation='relu', max_iter=1000),
)

for n_pc in range(3, 16):
    print(f"{n_pc} PCs")

    for model in models:
        print(f"{model_family} model: {model}")

        #try:

        # 1. Create run
        run = neptune.init_run(project="Combat4EEG/CombatVSModels",
                               api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiIzZDE5NGQ0NC1iNDQ1LTQ5OWYtOGI1OC03OTI0ZGE0ZGZkZWMifQ==",
                               capture_stdout=True,
                               source_files=["*.py"],
                               name="no name",
                               tags=[f"{len(dataset_combination)} datasets", "models eval", f"{n_pc} PCs", model_family],
                               )


        run['datasets/combination'] = str(list(dataset_combination))

        # 2. Configure run
        out_path = "./each_model"
        MMSE_criteria = (23, 27)
        MoCA_criteria = (18, 24)
        cov_age = True
        cov_gender = True
        cov_education = False
        cov_diagnosis = True

        # 4. Produce results
        make_results(run, out_path, list(dataset_combination), variant, cov_age, cov_gender, cov_education, cov_diagnosis, MMSE_criteria, MoCA_criteria, n_pc, model)

        # 5. Plot results
        make_images(run, out_path, list(dataset_combination), variant, model)

        # Run garbage collector
        print("Collected", str(gc.collect()), "unreachable objects.")

        run.stop()

        """
        except Exception as e:
            print("Error/Exception occurred. Stopping run.")
            print(e)
            run.stop()
        """

        # Clean up memory
        del run
        gc.collect()
