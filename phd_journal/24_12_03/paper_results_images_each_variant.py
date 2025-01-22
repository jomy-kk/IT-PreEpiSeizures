import itertools
import gc
import shutil

import neptune
from paper_results_each_variant import make_results
from paper_images_each_variant import make_images


class NoRun():
    class Item():
        def __init__(self):
            pass
        def upload(self, *args, **kwargs):
            pass
    def __init__(self):
        pass
    def __setitem__(self, key, value):
        pass
    def __getitem__(self, key):
        return NoRun.Item()
    def stop(self):
        pass


datasets = ['Izmir', 'Newcastle', 'Miltiadous', 'Istambul', 'BrainLat:CL', 'BrainLat:AR']
n_pcs = {  # n_datasets: [n_pcs]
    2: [2, 3, 4],
    3: [3, 4, 7],
    4: [6, 7, 9],
    5: [8, 11, 12],
    6: [8, 11, 12],
}
variants = ['neuroharmonize', ]#['none', 'neuroharmonize', 'neurocombat', 'original']

for n_datasets in (2, ):
    dataset_combinations = list(itertools.combinations(datasets, n_datasets))
    print(f"Number of combinations: {len(dataset_combinations)}")
    #print(dataset_combinations)
    #exit(0)
    #dataset_combinations = [('Izmir', 'Istambul', 'BrainLat:CL', 'BrainLat:AR'), ('Newcastle', 'Miltiadous', 'Istambul', 'BrainLat:CL'), ('Newcastle', 'Miltiadous', 'Istambul', 'BrainLat:AR'), ('Newcastle', 'Miltiadous', 'BrainLat:CL', 'BrainLat:AR'), ('Newcastle', 'Istambul', 'BrainLat:CL', 'BrainLat:AR'), ('Miltiadous', 'Istambul', 'BrainLat:CL', 'BrainLat:AR')]
    for dataset_combination in dataset_combinations:
        if dataset_combination == ('Izmir', 'Newcastle') or dataset_combination == ('Izmir', 'Miltiadous') or dataset_combination == ('Izmir', 'Istambul') or dataset_combination == ('Izmir', 'BrainLat:CL') or dataset_combination == ('Izmir', 'BrainLat:AR'):
            print("Skipping; already done.")
            continue
        print(dataset_combination)
        for n_pc in tuple(set(range(2, 16)) - set(n_pcs[n_datasets])):
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

                    # Run garbage collector
                    print("Collected", str(gc.collect()), "unreachable objects.")

                    run.stop()

                except Exception as e:
                    print("Error/Exception occurred. Stopping run.")
                    print(e)
                    run.stop()


                # Clean up memory
                del run
                gc.collect()

        # Delete all variant directory. Why? To not reuse the same results in the next iteration, when the combination changes.
        for variant in variants:
            out_path = f"./each_variation/{variant}"
            shutil.rmtree(out_path)
            print(f"Deleted {out_path}")
