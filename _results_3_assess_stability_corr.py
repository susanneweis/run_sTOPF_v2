import numpy as np
import pandas as pd

def main(base_path,res_path):

    # Map sex to numbers
    sex_mapping = {"male": 1, "female": 2}
    perc_values = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]

    true_sex_vecs = {}
    pred_sex_vecs = {}

    for reg_perc in perc_values:

        results_path = f"{base_path}/{res_path}/results_nn1/ind_classification_nn1"
        results_out_path = f"{base_path}/{res_path}"
        
        ind_class_name = f"classification_subjects_movies_corr_top_{reg_perc}perc.csv"
        ind_class_file =  f"{results_path}/{ind_class_name}"

        # Read CSV
        ind_class = pd.read_csv(ind_class_file)

        # Sort by movie, then subject
        ind_class = ind_class.sort_values(["movie", "subject"])

        # sex as numbers and the predicted sex as numbers:
        ind_class["sex_num"] = ind_class["sex"].map(sex_mapping)
        ind_class["class_num"] = ind_class["classification"].map(sex_mapping)

        true_sex_vecs[reg_perc]    = ind_class["sex_num"].to_numpy()
        pred_sex_vecs[reg_perc]   = ind_class["class_num"].to_numpy()

    avg_agreement_dict = {}

    for reg_perc in perc_values:  # 1..10
        v = pred_sex_vecs[reg_perc]

        sim_sum = 0.0
        n_other = 0

        for reg2 in perc_values:
            if reg2 == reg_perc:
                continue

            v2 = pred_sex_vecs[reg2]

            # similarity = fraction of equal entries
            sim = np.mean(v == v2)

            sim_sum += sim
            n_other += 1

        avg_agreement_dict[reg_perc] = sim_sum / n_other  # n_other should be 9

    df_avg = pd.DataFrame(
        {
            "perc": list(avg_agreement_dict.keys()),
            "avg_agreement": list(avg_agreement_dict.values()),
        }
    ).sort_values("perc")

    out_file = f"{results_out_path}/Stability_corr.csv"

    df_avg.to_csv(out_file, index=False)

    avg_neigh_agree_dict = {}

    for i in range(1, len(perc_values) - 1):
        left_perc = perc_values[i - 1]
        curr_perc  = perc_values[i]
        right_perc = perc_values[i + 1]

        sim = (np.mean(pred_sex_vecs[curr_perc] == pred_sex_vecs[left_perc]) + np.mean(pred_sex_vecs[curr_perc] == pred_sex_vecs[right_perc]))/2
        avg_neigh_agree_dict[curr_perc] = sim

    df_neigh_avg = pd.DataFrame(
        {
            "perc": list(avg_neigh_agree_dict.keys()),
            "avg_agreement": list(avg_neigh_agree_dict.values()),
        }
    ).sort_values("perc")

    out_file = f"{results_out_path}/Stability_neighbour_corr.csv"

    df_neigh_avg.to_csv(out_file, index=False)


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


# Execute script
if __name__ == "__main__":
    main()
