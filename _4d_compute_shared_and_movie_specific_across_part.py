import pandas as pd
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests
#from tqdm import tqdm
from scipy.stats import chi2
from scipy.stats import chi2

def main(base_path, proj, nn_mi, mov_prop):

    movies = list(mov_prop.keys())
    # movies = movies + ["concat"]
    # actual movies
    movies = movies[:-2]

    #load data and basic setup
    results_path = f"{base_path}/results_run_sTOPF_v2_data_{proj}/results_nn{nn_mi}"
    ind_ex_path = f"{results_path}/individual_expression_all_nn{nn_mi}.csv"
    ind_ex_data = pd.read_csv(ind_ex_path)

    # Optional: ensure categorical types
    ind_ex_data["subject"] = ind_ex_data["subject"].astype("category")
    ind_ex_data["movie"] = ind_ex_data["movie"].astype("category")
    ind_ex_data["region"] = ind_ex_data["region"].astype("category")

    #2. Mixed model for ONE region: Use statsmodels’s mixed-effects (Linear Mixed Effects) model:
    # Fixed effect: movie, Random intercept: subject

    # # pick one example region to start
    # example_region = ind_ex_data["region"].cat.categories[0]
    # # df_r = ind_ex_data[ind_ex_data["region"] == example_region]
    # df_r = ind_ex_data[
    #     (ind_ex_data["region"] == example_region) &
    #     (ind_ex_data["movie"].isin(movies))
    # ]

    # df_r["movie"] = df_r["movie"].cat.remove_unused_categories()
    # df_r["subject"] = df_r["subject"].cat.remove_unused_categories()
    # df_r["region"] = df_r["region"].cat.remove_unused_categories()
    # df_r = df_r.dropna(subset=["fem_vs_mal_corr", "movie", "subject"])

    # # mixed model: fem_vs_mal_corr ~ movie + (1 | subject)
    # md = smf.mixedlm("fem_vs_mal_corr ~ movie", df_r, groups=df_r["subject"])
    # m_full = md.fit(reml=False)
    # print(m_full.summary())

    # # This gives: Coefficients for each movie (relative to the reference movie)
    # # Standard errors, z-scores, p-values
    # # Variance of random subject effect, residual variance
    # # To test whether movie matters at all for this region, compare with a model without movie:

    # md_null = smf.mixedlm("fem_vs_mal_corr ~ 1", df_r, groups=df_r["subject"])
    # m_null = md_null.fit(reml=False)

    # # Likelihood ratio test for movie effect
    # lr_stat = 2 * (m_full.llf - m_null.llf)
    # df_diff = m_full.df_modelwc - m_null.df_modelwc  # degrees of freedom difference
    # p_lr = chi2.sf(lr_stat, df_diff)
    # print("Region:", example_region, "LR p-value for movie effect:", p_lr)

    # Interpretation for this region:
    # If p_lr is large → movie does not significantly improve fit → region is movie-general.
    # If p_lr is small (after multiple-comparison correction) → region shows movie-specific modulation.
    # The fixed-effect coefficients in m_full.params (those for movie[...]) tell you which movies are higher or lower than the reference movie.

    # 3. Run this across ALL regions
    # Now wrap this into a loop over regions:

    results = []

    #for region in df["region"].cat.categories:
    regions = ind_ex_data["region"].unique()
    for region in regions[:5]: 

        df_r = ind_ex_data[ind_ex_data["region"] == region]

        # skip regions with too few observations
        if df_r["subject"].nunique() < 3 or df_r["movie"].nunique() < 2:
            continue

        # full model
        try:
            m_full = smf.mixedlm("fem_vs_mal_corr ~ movie", df_r, groups=df_r["subject"]).fit(reml=False)
            m_null = smf.mixedlm("fem_vs_mal_corr ~ 1", df_r, groups=df_r["subject"]).fit(reml=False)
        except Exception as e:
            # sometimes models fail to converge; skip or log
            print("Region failed:", region, e)
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

    res_df = pd.DataFrame(results)

    # Then correct p-values across regions (e.g. FDR):

    res_df["p_lr_fdr"] = multipletests(res_df["p_lr"], method="fdr_bh")[1]

    # Now you can classify:
    # Movie-general regions: p_lr_fdr above your alpha (e.g. 0.05)
    # Movie-sensitive regions: p_lr_fdr <= 0.05

    # 4. Identify which movies a region is specific to
    # For regions with significant movie effect:
    # Look at the per-movie fixed effects from m_full.params.
    # Convert them to deviations around the region’s mean to get a “movie pattern”.
    # Example for one region:

    # get per-movie coefficients
    params = m_full.params  # includes intercept and movie dummies
    print(params)

    #You can store these per region:

    #movie_levels = ind_ex_data["movie"].cat.categories
    movie_levels = movies

    coef_rows = []

    #for region in df["region"].cat.categories:
    regions = ind_ex_data["region"].unique()
    for region in regions[:5]: 

        df_r = ind_ex_data[ind_ex_data["region"] == region]
        if df_r["subject"].nunique() < 3 or df_r["movie"].nunique() < 2:
            continue
        try:
            m_full = smf.mixedlm("fem_vs_mal_corr ~ movie", df_r, groups=df_r["subject"]).fit(reml=False)
            m_null = smf.mixedlm("fem_vs_mal_corr ~ 1", df_r, groups=df_r["subject"]).fit(reml=False)
        except Exception:
            continue

        lr_stat = 2 * (m_full.llf - m_null.llf)
        df_diff = m_full.df_modelwc - m_null.df_modelwc
        p_lr = chi2.sf(lr_stat, df_diff)

        # get fitted mean for each movie (region-specific)
        preds = []
        for mv in movie_levels:
            tmp = df_r.copy()
            tmp["movie"] = mv
            preds.append(m_full.predict(tmp).mean())
        for mv, pred in zip(movie_levels, preds):
            coef_rows.append({"region": region, "movie": mv, "mean_effect": pred, "p_lr": p_lr})
            
    coef_df = pd.DataFrame(coef_rows)
    coef_df.to_csv(f"{results_path}/movie_specificity_effects_nn{nn_mi}.csv", index=False)

    # Then, for each movie, you can z-score mean_effect across regions and pick regions where that movie stands out (positive or negative) as movie-specific.

# Execute script
if __name__ == "__main__":
    main()