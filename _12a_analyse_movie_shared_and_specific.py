import os
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests
from scipy.stats import chi2

def fit_movie_models(ind_ex_data, movies, max_regions=None):
    """
    For each region:
      - fit full mixed model: fem_vs_mal_corr ~ movie + (1|subject)
      - fit null mixed model: fem_vs_mal_corr ~ 1 + (1|subject)
      - likelihood ratio test for movie effect
      - estimate predicted mean effect for each movie

    Returns
    -------
    res_df : pd.DataFrame
        one row per region with LRT stats
    coef_df : pd.DataFrame
        one row per region x movie with predicted mean effect
    """
    results = []
    coef_rows = []

    regions = ind_ex_data["region"].unique()
    if max_regions is not None:
        regions = regions[:max_regions]

    for region in regions:
        df_r = ind_ex_data[ind_ex_data["region"] == region].copy()

        if df_r["subject"].nunique() < 3 or df_r["movie"].nunique() < 2:
            continue

        try:
            m_full = smf.mixedlm(
                "fem_vs_mal_corr ~ movie",
                df_r,
                groups=df_r["subject"]
            ).fit(reml=False)

            m_null = smf.mixedlm(
                "fem_vs_mal_corr ~ 1",
                df_r,
                groups=df_r["subject"]
            ).fit(reml=False)
        except Exception as e:
            print(f"Region failed: {region} | {e}")
            continue

        lr_stat = 2 * (m_full.llf - m_null.llf)
        df_diff = m_full.df_modelwc - m_null.df_modelwc
        p_lr = chi2.sf(lr_stat, df_diff)

        results.append({
            "region": region,
            "lr_stat": lr_stat,
            "df_diff": df_diff,
            "p_lr": p_lr
        })

        # predicted mean effect for each movie
        for mv in movies:
            tmp = df_r.copy()
            tmp["movie"] = mv
            pred = m_full.predict(tmp).mean()

            coef_rows.append({
                "region": region,
                "movie": mv,
                "mean_effect": pred,
                "p_lr": p_lr
            })

    res_df = pd.DataFrame(results)

    if not res_df.empty:
        res_df["p_lr_fdr"] = multipletests(res_df["p_lr"], method="fdr_bh")[1]
        res_df["movie_sensitive"] = res_df["p_lr_fdr"] <= 0.05
        res_df["neglog10_p_fdr"] = -np.log10(np.clip(res_df["p_lr_fdr"], 1e-300, None))

    coef_df = pd.DataFrame(coef_rows)

    if not res_df.empty and not coef_df.empty:
        coef_df = coef_df.merge(
            res_df[["region", "p_lr_fdr", "movie_sensitive", "neglog10_p_fdr"]],
            on="region",
            how="left"
        )

    return res_df, coef_df


def save_region_tables(res_df, coef_df, results_path, nn_mi):
    os.makedirs(results_path, exist_ok=True)

    res_df.to_csv(
        f"{results_path}/movie_specificity_nn{nn_mi}.csv",
        index=False
    )
    coef_df.to_csv(
        f"{results_path}/movie_specificity_per_movie_nn{nn_mi}.csv",
        index=False
    )

def main(base_path, proj, nn_mi, mov_prop, max_regions):
    movies = list(mov_prop.keys())
    # actual movies
    movies = movies[:-2]   # excluding last two, as in your script

    results_path = f"{base_path}/results_run_sTOPF_v2_data_{proj}/results_nn{nn_mi}"
    ind_ex_path = f"{results_path}/individual_expression_all_nn{nn_mi}.csv"

    results_out_path = f"{results_path}/movie_sensitivity_nn{nn_mi}"
    os.makedirs(results_out_path, exist_ok=True)

    ind_ex_data = pd.read_csv(ind_ex_path)

    ind_ex_data["subject"] = ind_ex_data["subject"].astype("category")
    ind_ex_data["movie"] = ind_ex_data["movie"].astype("category")
    ind_ex_data["region"] = ind_ex_data["region"].astype("category")

    res_df, coef_df = fit_movie_models(
        ind_ex_data=ind_ex_data,
        movies=movies,
        max_regions=max_regions
    )

    save_region_tables(res_df, coef_df, results_out_path, nn_mi)