import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests
from scipy.stats import chi2
from _util_glass_brains import create_glassbrains


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


def make_brain_maps(res_df, coef_df, results_path, nn_mi,
                    atlas_path, roi_names):
    """
    Uses your own create_glassbrains(...) helper.
    Assumes it can take a dataframe with columns:
      - region
      - value column to plot
    """

    # 1) Movie-sensitive significance map
    sig_map = res_df[["region", "neglog10_p_fdr"]].copy()
    sig_file = f"{results_path}/brainmap_movie_sensitivity_nn{nn_mi}.csv"
    sig_map.to_csv(sig_file, index=False)

    create_glassbrains(sig_file,"neglog10_p_fdr","region",roi_names,atlas_path,"Movie sensitivity (-log10 FDR p)",results_path,f"brainmap_movie_sensitivity_nn{nn_mi}","continuous")

    # 2) Binary movie-general map
    gen_map = res_df[["region", "movie_sensitive"]].copy()
    gen_map["movie_general"] = (~gen_map["movie_sensitive"]).astype(int)
    gen_file = f"{results_path}/brainmap_movie_general_nn{nn_mi}.csv"
    gen_map[["region", "movie_general"]].to_csv(gen_file, index=False)

    create_glassbrains(gen_file,"movie_general","region",roi_names,atlas_path,"Movie-general regions",results_path,f"brainmap_movie_general_nn{nn_mi}","discrete")

    # 3) One map per movie: predicted effect
    for mv in coef_df["movie"].unique():
        mv_df = coef_df.loc[coef_df["movie"] == mv, ["region", "mean_effect"]].copy()
        mv_file = f"{results_path}/brainmap_{mv}_mean_effect_nn{nn_mi}.csv"
        mv_df.to_csv(mv_file, index=False)

        create_glassbrains(mv_file,"mean_effect","region",roi_names,atlas_path,f"{mv}: predicted mean effect",results_path,f"brainmap_{mv}_mean_effect_nn{nn_mi}","continuous")


def plot_top_sensitive_regions(res_df, results_path, nn_mi, top_n=20):
    if res_df.empty:
        return

    plot_df = res_df.sort_values("neglog10_p_fdr", ascending=False).head(top_n).copy()

    plt.figure(figsize=(10, max(6, top_n * 0.35)))
    plt.barh(plot_df["region"].astype(str), plot_df["neglog10_p_fdr"])
    plt.gca().invert_yaxis()
    plt.xlabel("-log10(FDR p)")
    plt.ylabel("Region")
    plt.title(f"Top {top_n} movie-sensitive regions")
    plt.tight_layout()
    plt.savefig(
        f"{results_path}/barplot_top_movie_sensitive_regions_nn{nn_mi}.png",
        dpi=300,
        bbox_inches="tight"
    )
    plt.close()


def plot_region_movie_profiles(coef_df, res_df, results_path, nn_mi,
                               regions_to_plot=None, top_n=6):
    """
    If regions_to_plot is None:
      take top_n most movie-sensitive regions from res_df
    """

    if coef_df.empty or res_df.empty:
        return

    if regions_to_plot is None:
        regions_to_plot = (
            res_df.sort_values("neglog10_p_fdr", ascending=False)
                  .head(top_n)["region"]
                  .tolist()
        )

    for region in regions_to_plot:
        tmp = coef_df.loc[coef_df["region"] == region].copy()
        if tmp.empty:
            continue

        plt.figure(figsize=(8, 4))
        plt.bar(tmp["movie"].astype(str), tmp["mean_effect"])
        plt.axhline(0, linewidth=1)
        plt.ylabel("Predicted mean effect")
        plt.xlabel("Movie")
        plt.title(f"Movie profile: {region}")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        safe_region = str(region).replace("/", "_").replace(" ", "_")
        plt.savefig(
            f"{results_path}/barplot_movie_profile_{safe_region}_nn{nn_mi}.png",
            dpi=300,
            bbox_inches="tight"
        )
        plt.close()


def plot_movie_heatmap(coef_df, res_df, results_path, nn_mi, top_n=30):
    """
    Heatmap: rows = most movie-sensitive regions, cols = movies, values = mean_effect
    """
    if coef_df.empty or res_df.empty:
        return

    top_regions = (
        res_df.sort_values("neglog10_p_fdr", ascending=False)
              .head(top_n)["region"]
              .tolist()
    )

    heat_df = coef_df[coef_df["region"].isin(top_regions)].copy()
    pivot = heat_df.pivot(index="region", columns="movie", values="mean_effect")

    plt.figure(figsize=(1.2 * max(4, pivot.shape[1]), 0.35 * max(8, pivot.shape[0])))
    plt.imshow(pivot.values, aspect="auto")
    plt.colorbar(label="Predicted mean effect")
    plt.xticks(range(len(pivot.columns)), pivot.columns, rotation=45, ha="right")
    plt.yticks(range(len(pivot.index)), pivot.index)
    plt.title(f"Movie effects across top {top_n} movie-sensitive regions")
    plt.tight_layout()
    plt.savefig(
        f"{results_path}/heatmap_movie_effects_top_regions_nn{nn_mi}.png",
        dpi=300,
        bbox_inches="tight"
    )
    plt.close()

    pivot.to_csv(f"{results_path}/heatmap_movie_effects_top_regions_nn{nn_mi}.csv")


def main(base_path, proj, nn_mi, mov_prop, atlas_path, roi_names,max_regions):
    movies = list(mov_prop.keys())
    # actual movies
    movies = movies[:-2]   # excluding last two, as in your script

    results_path = f"{base_path}/results_run_sTOPF_v2_data_{proj}/results_nn{nn_mi}"
    ind_ex_path = f"{results_path}/individual_expression_all_nn{nn_mi}.csv"

    results_out_path = f"{results_path}/movie_sensitivity_nn{nn_mi}"
    os.makedirs(results_out_path, exist_ok=True)
    results_glass_brains_out_path = f"{results_path}/movie_sensitivity_nn{nn_mi}/glass_brains"
    os.makedirs(results_glass_brains_out_path, exist_ok=True)
    results_data_out_path = f"{results_path}/movie_sensitivity_nn{nn_mi}/tables"
    os.makedirs(results_data_out_path, exist_ok=True)

    ind_ex_data = pd.read_csv(ind_ex_path)

    ind_ex_data["subject"] = ind_ex_data["subject"].astype("category")
    ind_ex_data["movie"] = ind_ex_data["movie"].astype("category")
    ind_ex_data["region"] = ind_ex_data["region"].astype("category")

    res_df, coef_df = fit_movie_models(
        ind_ex_data=ind_ex_data,
        movies=movies,
        max_regions=max_regions
    )

    save_region_tables(res_df, coef_df, results_data_out_path, nn_mi)

    make_brain_maps(
        res_df=res_df,
        coef_df=coef_df,
        results_path=results_glass_brains_out_path,
        nn_mi=nn_mi,
        atlas_path=atlas_path,
        roi_names=roi_names,
    )

    plot_top_sensitive_regions(res_df, results_data_out_path, nn_mi, top_n=20)
    plot_region_movie_profiles(coef_df, res_df, results_data_out_path, nn_mi, top_n=6)
    plot_movie_heatmap(coef_df, res_df, results_data_out_path, nn_mi, top_n=30)