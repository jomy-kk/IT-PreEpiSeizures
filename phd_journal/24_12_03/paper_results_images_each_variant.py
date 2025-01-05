import itertools
import gc

import neptune
from paper_results_each_variant import make_results
from paper_images_each_variant import make_images

datasets = ['Izmir', 'Newcastle', 'Miltiadous', 'Istambul', 'BrainLat:CL', 'BrainLat:AR']
n_pcs = range(2, 16) # 2 ... 15
variants = ['none', 'neuroharmonize']

for n_datasets in range(5, 7):
    dataset_combinations = list(itertools.combinations(datasets, n_datasets))
    print(f"Number of combinations: {len(dataset_combinations)}")
    #print(dataset_combinations)
    #exit(0)
    #dataset_combinations = [('Izmir', 'Istambul', 'BrainLat:CL', 'BrainLat:AR'), ('Newcastle', 'Miltiadous', 'Istambul', 'BrainLat:CL'), ('Newcastle', 'Miltiadous', 'Istambul', 'BrainLat:AR'), ('Newcastle', 'Miltiadous', 'BrainLat:CL', 'BrainLat:AR'), ('Newcastle', 'Istambul', 'BrainLat:CL', 'BrainLat:AR'), ('Miltiadous', 'Istambul', 'BrainLat:CL', 'BrainLat:AR')]
    for dataset_combination in dataset_combinations:
        print(dataset_combination)
        for n_pc in n_pcs:
            print(f"{n_pc} PCs")
            for variant in variants:
                print(f"{variant} variant")

                try:

                    # 1. Create run
                    run = neptune.init_run(project="Combat4EEG/CombatManuscript",
                                           api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiIzZDE5NGQ0NC1iNDQ1LTQ5OWYtOGI1OC03OTI0ZGE0ZGZkZWMifQ==",
                                           capture_stdout=True,
                                           source_files=["*.py"],
                                           name="no name",
                                           tags=[f"{len(dataset_combination)} datasets", "mmse gap 24-26",] + list(dataset_combination),
                                           )

                    run['datasets/combination'] = str(list(dataset_combination))

                    # 2. Configure run
                    out_path = "./each_variation"
                    cov_age = True
                    cov_gender = True
                    cov_education = False
                    cov_diagnosis = True
                    MMSE_criteria = (23, 27)
                    MoCA_criteria = (18, 24)

                    # 3. Produce results
                    make_results(run, out_path, list(dataset_combination), variant, cov_age, cov_gender, cov_education, cov_diagnosis, MMSE_criteria, MoCA_criteria, n_pc)

                    # 4. Plot results
                    make_images(run, out_path, list(dataset_combination), variant)

                    run.stop()

                except Exception as e:
                    print("Error/Exception occurred. Stopping run.")
                    print(e)
                    run.stop()
                    exit(-1)


                # Clean up memory
                del run
                gc.collect()

