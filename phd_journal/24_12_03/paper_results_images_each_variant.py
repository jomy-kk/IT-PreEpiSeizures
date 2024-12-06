import neptune
from paper_results_each_variant import make_results
from paper_images_each_variant import make_images

# 1. Create run (use global)
run = neptune.init_run(project="Combat4EEG/CombatManuscript",
                       api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiIzZDE5NGQ0NC1iNDQ1LTQ5OWYtOGI1OC03OTI0ZGE0ZGZkZWMifQ==",
                       capture_stdout=True,
                       source_files=["*.py"],
                       name="no name",
                       tags=["4 datasets", "mmse gap 24-26", "simple discriminant"],
                       )

# 2. Configure run
out_path = "./each_variation"
harmonization_method = "none"
cov_age = True
cov_gender = True
cov_education = False
cov_diagnosis = True
MMSE_criteria = (23, 27)
MoCA_criteria = (18, 24)

# 3. Produce results
make_results(run, out_path, harmonization_method, cov_age, cov_gender, cov_education, cov_diagnosis, MMSE_criteria, MoCA_criteria)

# 4. Plot results
make_images(run, out_path, harmonization_method)

