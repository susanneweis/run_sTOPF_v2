import numpy as np
import pandas as pd

base_path =  "/Users/sweis/Data/Arbeit/Juseless/data/project/brainvar_sexdiff_movies" 
proj = "v2"

# Map sex to numbers
sex_mapping = {"male": 1, "female": 2}

results_path = f"{base_path}/results_run_sTOPF_{proj}/results_nn{nn_mi}/ind_classification"
results_out_path = f"{base_path}/results_run_sTOPF_{proj}"

true_sex_vecs = {}
pred_sex_vecs = {}

for nn_mi in range(1, 11):
    ind_class_name = f"classification_subjects_movies_nn{nn_mi}_top_100perc.csv"
    ind_class_file =  f"{results_path}/{ind_class_name}"

    # Read CSV
    ind_class = pd.read_csv(ind_class_file)

    # Sort by movie, then subject
    ind_class = ind_class.sort_values(["movie", "subject"])

    # sex as numbers and the predicted sex as numbers:
    ind_class["sex_num"] = ind_class["sex"].map(sex_mapping)
    ind_class["class_num"] = ind_class["classification"].map(sex_mapping)

    true_sex_vecs[nn_mi]    = ind_class["sex_num"].to_numpy()
    pred_sex_vecs[nn_mi]   = ind_class["class_num"].to_numpy()

avg_agreement_dict = {}

for nn in range(1, 11):  # 1..10
    v = pred_sex_vecs[nn]

    sim_sum = 0.0
    n_other = 0

    for nn2 in range(1, 11):
        if nn2 == nn:
            continue

        v2 = pred_sex_vecs[nn2]

        # similarity = fraction of equal entries
        sim = np.mean(v == v2)

        sim_sum += sim
        n_other += 1

    avg_agreement_dict[nn] = sim_sum / n_other  # n_other should be 9

df_avg = pd.DataFrame(
    {
        "nn_mi": list(avg_agreement_dict.keys()),
        "avg_agreement": list(avg_agreement_dict.values()),
    }
).sort_values("nn_mi")

out_file = f"{results_out_path}/Stability_nn_mi.csv"

df_avg.to_csv(out_file, index=False)

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

