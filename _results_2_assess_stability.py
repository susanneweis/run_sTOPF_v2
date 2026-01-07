import numpy as np
import pandas as pd

base_path =  "/Users/sweis/Data/Arbeit/Juseless/data/project/brainvar_sexdiff_movies" 
proj = "v2"

nn_mi = 3 

results_out_path = f"{base_path}/results_run_sTOPF_{proj}/results_nn{nn_mi}/ind_classification"

ind_class_name = f"classification_subjects_movies_nn{nn_mi}_top_100perc.csv"
ind_class_file =  f"{results_out_path}/{ind_class_name}"

# Read CSV
ind_class = pd.read_csv(ind_class_file)

# Sort by movie, then subject
ind_class = ind_class.sort_values(["movie", "subject"])

# Map sex to numbers
sex_mapping = {"male": 1, "female": 2}

# sex as numbers and the predicted sex as numbers:
ind_class["sex_num"] = ind_class["sex"].map(sex_mapping)
ind_class["class_num"] = ind_class["classification"].map(sex_mapping)

true_sex_vec    = ind_class["sex_num"].to_numpy()
pred_sex_vec    = ind_class["class_num"].to_numpy()

# pairwise agreement
#agree_3_5 = (labels_3 == labels_5).mean()
#agree_3_7 = (labels_3 == labels_7).mean()
#agree_5_7 = (labels_5 == labels_7).mean()

# “average agreement with others”
#stab_3 = (agree_3_5 + agree_3_7) / 2
#stab_5 = (agree_3_5 + agree_5_7) / 2
#stab_7 = (agree_3_7 + agree_5_7) / 2

# The nn with the highest stab_nn is the one whose male/female decision changes least when 
# you switch to another nn, i.e. the most stable.
# If all stab_nn are very similar and high (e.g. > 0.95), just pick the smallest nn (3) or stick 
# with the default for simplicity and report that results are robust across 

