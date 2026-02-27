import pandas as pd

def main(base_path,proj,nn_mi,movies_properties):

    results_out_path = f"{base_path}/results_run_sTOPF_v2_data_{proj}/results_nn{nn_mi}"

    # Change this later 
    ind_expr_path = f"{results_out_path}/individual_expression_all_nn{nn_mi}.csv"
    ind_expr = pd.read_csv(ind_expr_path)

    movies = list(movies_properties.keys())
    movies = movies + ["concat"]

    regions = ind_expr["region"].astype(str).drop_duplicates().tolist()
    
    # res_summary = []

    ind_expr["class_corr_correct"] = (
        ((ind_expr["sex"] == "female") & (ind_expr["fem_vs_mal_corr"] >= 0)) |
        ((ind_expr["sex"] == "male") & (ind_expr["fem_vs_mal_corr"] < 0))
    )

    ind_expr["class_regr_correct"] = (
        ((ind_expr["sex"] == "female") & (ind_expr["fem_vs_mal_regr"] >= 0)) |
        ((ind_expr["sex"] == "male") & (ind_expr["fem_vs_mal_regr"] < 0))
    )
    # ind_expr.to_csv(f"{results_out_path}/correct_classification_fem_vs_mal_corr.csv", index=False)

    ind_expr["class_mi_correct"] = (
        ((ind_expr["sex"] == "female") & (ind_expr["fem_mi"] >= ind_expr["mal_mi"] )) |
        ((ind_expr["sex"] == "male") & (ind_expr["fem_mi"] < ind_expr["mal_mi"]))
    )
    ind_expr.to_csv(f"{results_out_path}/correct_classifications_corr_regr_mi.csv", index=False)

    movie_class_summary = []

    for curr_mov in movies:
        
        mv_class = ind_expr.loc[ind_expr["movie"] == curr_mov, ["sex","class_corr_correct","class_regr_correct","class_mi_correct"]].reset_index(drop=True)

        mv_class_fem = mv_class.loc[mv_class["sex"] == "female", ["sex","class_corr_correct","class_regr_correct","class_mi_correct"]].reset_index(drop=True)
        mv_class_mal = mv_class.loc[mv_class["sex"] == "male", ["sex","class_corr_correct","class_regr_correct","class_mi_correct"]].reset_index(drop=True)
        nr_fem = len(mv_class_fem)
        nr_mal = len(mv_class_mal)

        count_true_fem = mv_class_fem["class_corr_correct"].sum()
        count_true_mal = mv_class_mal["class_corr_correct"].sum()

        count_true_fem_sim = mv_class_fem["class_regr_correct"].sum()
        count_true_mal_sim = mv_class_mal["class_regr_correct"].sum()
    
        count_true_fem_mi = mv_class_fem["class_mi_correct"].sum()
        count_true_mal_mi = mv_class_mal["class_mi_correct"].sum()

        movie_class_summary.append({"movie": curr_mov, "female accuracy fem_vs_mal_corr": count_true_fem/nr_fem, "male accuracy fem_vs_mal_corr": count_true_mal/nr_mal, "female accuracy fem_regr": count_true_fem_sim/nr_fem, "male accuracy fem_regr": count_true_mal_sim/nr_mal, "female accuracy fem_vs_mal_mi": count_true_fem_mi/nr_fem, "male accuracy fem_vs_mal_mi": count_true_mal_mi/nr_mal})

    movie_class_summary_df = pd.DataFrame(movie_class_summary)
    movie_class_summary_df.to_csv(f"{results_out_path}/correct_classification_per_movie_nn{nn_mi}.csv", index=False)

    region_class_summary = []
    for curr_reg in regions:
        reg_class = ind_expr.loc[ind_expr["region"] == curr_reg, ["sex","class_corr","class_corr_sim","class_corr_mi"]].reset_index(drop=True)

        reg_class_fem = reg_class.loc[reg_class["sex"] == "female", ["sex","class_corr","class_corr_sim","class_corr_mi"]].reset_index(drop=True)
        reg_class_mal = reg_class.loc[reg_class["sex"] == "male", ["sex","class_corr","class_corr_sim","class_corr_mi"]].reset_index(drop=True)
        nr_fem = len(reg_class_fem)
        nr_mal = len(reg_class_mal)

        count_true_fem_r = reg_class_fem["class_corr"].sum()
        count_true_mal_r = reg_class_mal["class_corr"].sum()

        count_true_fem_r_sim = reg_class_fem["class_corr_sim"].sum()
        count_true_mal_r_sim = reg_class_mal["class_corr_sim"].sum()

        count_true_fem_r_mi = reg_class_fem["class_corr_mi"].sum()
        count_true_mal_r_mi = reg_class_mal["class_corr_mi"].sum()

        region_class_summary.append({"region": curr_reg, "female corr fem_vs_mal_corr": count_true_fem_r/nr_fem, "male corr fem_vs_mal_corr": count_true_mal_r/nr_mal, "female corr fem_sim": count_true_fem_r_sim/nr_fem, "male corr fem_sim": count_true_mal_r_sim/nr_mal, "female corr mi": count_true_fem_r_mi/nr_fem, "male corr mi": count_true_mal_r_mi/nr_mal})

    region_class_summary_df = pd.DataFrame(region_class_summary)
    region_class_summary_df.to_csv(f"{results_out_path}/correct_classification_per_region_nn{nn_mi}.csv", index=False)

    #act_movies = ["dd", "s", "dps", "fg", "dmw", "lib", "tgtbtu"]
    act_movies = movies[:-2]

    act_mv_region_class_summary = []
    for curr_reg in regions:
        reg_class = ind_expr.loc[ind_expr["region"] == curr_reg, ["sex","movie","class_corr","class_corr_sim","class_corr_mi"]].reset_index(drop=True)
        reg_class = reg_class[reg_class["movie"].isin(act_movies)]

        reg_class_fem = reg_class.loc[reg_class["sex"] == "female", ["sex","movie","class_corr","class_corr_sim","class_corr_mi"]].reset_index(drop=True)
        reg_class_mal = reg_class.loc[reg_class["sex"] == "male", ["sex","movie","class_corr","class_corr_sim","class_corr_mi"]].reset_index(drop=True)
        nr_fem = len(reg_class_fem)
        nr_mal = len(reg_class_mal)

        count_true_fem_r = reg_class_fem["class_corr"].sum()
        count_true_mal_r = reg_class_mal["class_corr"].sum()

        count_true_fem_r_sim = reg_class_fem["class_corr_sim"].sum()
        count_true_mal_r_sim = reg_class_mal["class_corr_sim"].sum()
    
        count_true_fem_r_mi = reg_class_fem["class_corr_mi"].sum()
        count_true_mal_r_mi = reg_class_mal["class_corr_mi"].sum()

        act_mv_region_class_summary.append({"region": curr_reg, "female corr fem_vs_mal_corr": count_true_fem_r/nr_fem, "male corr fem_vs_mal_corr": count_true_mal_r/nr_mal, "female corr fem_sim": count_true_fem_r_sim/nr_fem, "male corr fem_sim": count_true_mal_r_sim/nr_mal, "female corr mi": count_true_fem_r_mi/nr_fem, "male corr mi": count_true_mal_r_mi/nr_mal})

    act_mv_region_class_summary_df = pd.DataFrame(act_mv_region_class_summary)
    act_mv_region_class_summary_df.to_csv(f"{results_out_path}/correct_classification_per_region_no_rest_nn{nn_mi}.csv", index=False)

    mv_reg_class_summary = []
    for curr_reg in regions:
        reg_class = ind_expr.loc[ind_expr["region"] == curr_reg, ["sex","movie","class_corr","class_corr_sim","class_corr_mi"]].reset_index(drop=True)

        for curr_mov in movies: 

            reg_class = ind_expr.loc[ind_expr["movie"] == curr_mov, ["sex","class_corr","class_corr_sim","class_corr_mi"]].reset_index(drop=True)

            reg_class_fem = reg_class.loc[reg_class["sex"] == "female", ["sex","class_corr","class_corr_sim","class_corr_mi"]].reset_index(drop=True)
            reg_class_mal = reg_class.loc[reg_class["sex"] == "male", ["sex","class_corr","class_corr_sim","class_corr_mi"]].reset_index(drop=True)
            nr_fem = len(reg_class_fem)
            nr_mal = len(reg_class_mal)

            count_true_fem_r = reg_class_fem["class_corr"].sum()
            count_true_mal_r = reg_class_mal["class_corr"].sum()

            count_true_fem_r_sim = reg_class_fem["class_corr_sim"].sum()
            count_true_mal_r_sim = reg_class_mal["class_corr_sim"].sum()

            count_true_fem_r_mi = reg_class_fem["class_corr_mi"].sum()
            count_true_mal_r_mi = reg_class_mal["class_corr_mi"].sum()

            mv_reg_class_summary.append({"region": curr_reg, "movie": curr_mov, "female corr fem_vs_mal_corr": count_true_fem_r/nr_fem, "male corr fem_vs_mal_corr": count_true_mal_r/nr_mal, "female corr fem_sim": count_true_fem_r_sim/nr_fem, "male corr fem_sim": count_true_mal_r_sim/nr_mal,  "female corr mi": count_true_fem_r_mi/nr_fem, "male corr mi": count_true_mal_r_mi/nr_mal})

    mv_reg_class_summary_df = pd.DataFrame(mv_reg_class_summary)
    mv_reg_class_summary_df.to_csv(f"{results_out_path}/correct_classification_per_region_per_movie_nn{nn_mi}.csv", index=False)

    ind_expr = pd.read_csv(ind_expr_path)
    
    # Aggregate per subject+movie
    out = (
        ind_expr.groupby(["subject", "sex", "movie"], as_index=False)
        .agg(
            mean_fem_vs_mal_corr=("fem_vs_mal_corr", "mean"),                 # NaNs ignored by default
            mean_fem_vs_mal_regr=("fem_vs_mal_regr", "mean"),                 # NaNs ignored by default
            neg_perc_fem_vs_mal_corr =("fem_vs_mal_corr", lambda s: (s < 0).sum()/436), # NaNs don't count as negative
            neg_perc_fem_vs_mal_regr =("fem_vs_mal_regr", lambda s: (s < 0).sum()/436) # NaNs don't count as negative
        )
        .sort_values(["subject", "movie"])
    )

    # Save result
    out.to_csv(f"{results_out_path}/subject_movie_summary_nn{nn_mi}.csv", index=False)


# Execute script
if __name__ == "__main__":
    main()
