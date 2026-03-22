import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests
from scipy.stats import chi2
from _util_glass_brains import create_glassbrains


def make_brain_maps(res_df, coef_df, results_g_path, results_t_path, nn_mi,
                    atlas_path, roi_names):
    """
    Uses your own create_glassbrains(...) helper.
    Assumes it can take a dataframe with columns:
      - region
      - value column to plot
    """

    # 1) Movie-sensitive significance map
    sig_map = res_df[["region", "neglog10_p_fdr"]].copy()
    sig_file = f"{results_t_path}/brainmap_movie_sensitivity_nn{nn_mi}.csv"
    sig_map.to_csv(sig_file, index=False)

    create_glassbrains(sig_file,"neglog10_p_fdr","region",roi_names,atlas_path,"Movie sensitivity (-log10 FDR p)",results_g_path,f"brainmap_movie_sensitivity_nn{nn_mi}","continuous")

    # 2) Binary movie-general map
    gen_map = res_df[["region", "movie_sensitive"]].copy()
    gen_map["movie_general"] = (~gen_map["movie_sensitive"]).astype(int)
    gen_file = f"{results_t_path}/brainmap_movie_general_nn{nn_mi}.csv"
    gen_map[["region", "movie_general"]].to_csv(gen_file, index=False)

    create_glassbrains(gen_file,"movie_general","region",roi_names,atlas_path,"Movie-general regions",results_g_path,f"brainmap_movie_general_nn{nn_mi}","discrete")

    # 3) One map per movie: predicted effect
    for mv in coef_df["movie"].unique():
        mv_df = coef_df.loc[coef_df["movie"] == mv, ["region", "mean_effect"]].copy()
        mv_file = f"{results_t_path}/brainmap_{mv}_mean_effect_nn{nn_mi}.csv"
        mv_df.to_csv(mv_file, index=False)

        create_glassbrains(mv_file,"mean_effect","region",roi_names,atlas_path,f"{mv}: predicted mean effect",results_g_path,f"brainmap_{mv}_mean_effect_nn{nn_mi}","continuous")


# alt version with top n regions
# def plot_top_sensitive_regions(res_df, results_path, nn_mi, top_n=20):
#     if res_df.empty:
#         return

#     plot_df = res_df.sort_values("neglog10_p_fdr", ascending=False).head(top_n).copy()

#     plt.figure(figsize=(10, max(6, top_n * 0.35)))
#     plt.barh(plot_df["region"].astype(str), plot_df["neglog10_p_fdr"])
#     plt.gca().invert_yaxis()
#     plt.xlabel("-log10(FDR p)")
#     plt.ylabel("Region")
#     plt.title(f"Top {top_n} movie-sensitive regions")
#     plt.tight_layout()
#     plt.savefig(
#         f"{results_path}/barplot_top_movie_sensitive_regions_nn{nn_mi}.png",
#         dpi=300,
#         bbox_inches="tight"
#     )
#     plt.close()

def plot_top_sensitive_regions(res_df, results_path, nn_mi):
    if res_df.empty:
        return

    plot_df = res_df[res_df["movie_sensitive"] == True].copy()

    if plot_df.empty:
        return

    plot_df = plot_df.sort_values("neglog10_p_fdr", ascending=False)

    plt.figure(figsize=(10, max(6, len(plot_df) * 0.35)))
    plt.barh(plot_df["region"].astype(str), plot_df["neglog10_p_fdr"])
    plt.gca().invert_yaxis()
    plt.xlabel("-log10(FDR p)")
    plt.ylabel("Region")
    plt.title("All movie-sensitive regions")
    plt.tight_layout()
    plt.savefig(
        f"{results_path}/barplot_movie_sensitive_regions_nn{nn_mi}.png",
        dpi=300,
        bbox_inches="tight"
    )
    plt.close()

# alt version with top n regions
# def plot_region_movie_profiles(coef_df, res_df, results_path, nn_mi,
#                                regions_to_plot=None, top_n=6):
#     """
#     If regions_to_plot is None:
#       take top_n most movie-sensitive regions from res_df
#     """

#     if coef_df.empty or res_df.empty:
#         return

#     if regions_to_plot is None:
#         regions_to_plot = (
#             res_df.sort_values("neglog10_p_fdr", ascending=False)
#                   .head(top_n)["region"]
#                   .tolist()
#         )

#     for region in regions_to_plot:
#         tmp = coef_df.loc[coef_df["region"] == region].copy()
#         if tmp.empty:
#             continue

#         plt.figure(figsize=(8, 4))
#         plt.bar(tmp["movie"].astype(str), tmp["mean_effect"])
#         plt.axhline(0, linewidth=1)
#         plt.ylabel("Predicted mean effect")
#         plt.xlabel("Movie")
#         plt.title(f"Movie profile: {region}")
#         plt.xticks(rotation=45, ha="right")
#         plt.tight_layout()
#         safe_region = str(region).replace("/", "_").replace(" ", "_")
#         plt.savefig(
#             f"{results_path}/barplot_movie_profile_{safe_region}_nn{nn_mi}.png",
#             dpi=300,
#             bbox_inches="tight"
#         )
#         plt.close()

def plot_region_movie_profiles(coef_df, res_df, results_path, nn_mi,
                               regions_to_plot=None):
    """
    If regions_to_plot is None:
      take all movie-sensitive regions from res_df
    """

    if coef_df.empty or res_df.empty:
        return

    if regions_to_plot is None:
        regions_to_plot = (
            res_df.loc[res_df["movie_sensitive"] == True, "region"]
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

# alt version with top n regions
# def plot_movie_heatmap(coef_df, res_df, results_path, nn_mi, top_n=30):
#     """
#     Heatmap: rows = most movie-sensitive regions, cols = movies, values = mean_effect
#     """
#     if coef_df.empty or res_df.empty:
#         return

#     top_regions = (
#         res_df.sort_values("neglog10_p_fdr", ascending=False)
#               .head(top_n)["region"]
#               .tolist()
#     )

#     heat_df = coef_df[coef_df["region"].isin(top_regions)].copy()
#     pivot = heat_df.pivot(index="region", columns="movie", values="mean_effect")

#     plt.figure(figsize=(1.2 * max(4, pivot.shape[1]), 0.35 * max(8, pivot.shape[0])))
#     plt.imshow(pivot.values, aspect="auto")
#     plt.colorbar(label="Predicted mean effect")
#     plt.xticks(range(len(pivot.columns)), pivot.columns, rotation=45, ha="right")
#     plt.yticks(range(len(pivot.index)), pivot.index)
#     plt.title(f"Movie effects across top {top_n} movie-sensitive regions")
#     plt.tight_layout()
#     plt.savefig(
#         f"{results_path}/heatmap_movie_effects_top_regions_nn{nn_mi}.png",
#         dpi=300,
#         bbox_inches="tight"
#     )
#     plt.close()

#     pivot.to_csv(f"{results_path}/heatmap_movie_effects_top_regions_nn{nn_mi}.csv")

def plot_movie_heatmap(coef_df, res_df, results_path, nn_mi):
    """
    Heatmap: rows = movie-sensitive regions, cols = movies, values = mean_effect
    """
    if coef_df.empty or res_df.empty:
        return

    selected_regions = (
        res_df.loc[res_df["movie_sensitive"] == True, "region"]
              .tolist()
    )

    if len(selected_regions) == 0:
        return

    heat_df = coef_df[coef_df["region"].isin(selected_regions)].copy()
    pivot = heat_df.pivot(index="region", columns="movie", values="mean_effect")

    plt.figure(figsize=(1.2 * max(4, pivot.shape[1]), 0.35 * max(8, pivot.shape[0])))
    plt.imshow(pivot.values, aspect="auto")
    plt.colorbar(label="Predicted mean effect")
    plt.xticks(range(len(pivot.columns)), pivot.columns, rotation=45, ha="right")
    plt.yticks(range(len(pivot.index)), pivot.index)
    plt.title("Movie effects across movie-sensitive regions")
    plt.tight_layout()
    plt.savefig(
        f"{results_path}/heatmap_movie_effects_movie_sensitive_regions_nn{nn_mi}.png",
        dpi=300,
        bbox_inches="tight"
    )
    plt.close()

    pivot.to_csv(
        f"{results_path}/heatmap_movie_effects_movie_sensitive_regions_nn{nn_mi}.csv"
    )


def plot_movie_heatmap_all_regions(coef_df, results_path, nn_mi, roi_names):
    """
    Heatmap: rows = all regions in roi_names order,
             cols = movies,
             values = mean_effect
    """
    if coef_df.empty:
        return

    # Keep only regions that are in roi_names
    heat_df = coef_df[coef_df["region"].isin(roi_names)].copy()

    if heat_df.empty:
        return

    # Create region x movie matrix
    pivot = heat_df.pivot(index="region", columns="movie", values="mean_effect")

    # Reorder rows to match ROI_names.csv exactly
    pivot = pivot.reindex(roi_names)

    plt.figure(figsize=(1.2 * max(4, pivot.shape[1]), 0.35 * max(8, pivot.shape[0])))
    plt.imshow(pivot.values, aspect="auto")
    plt.colorbar(label="Predicted mean effect")
    plt.xticks(range(len(pivot.columns)), pivot.columns, rotation=45, ha="right")
    plt.yticks(range(len(pivot.index)), pivot.index)
    plt.title("Movie effects across all regions")
    plt.tight_layout()
    plt.savefig(
        f"{results_path}/heatmap_movie_effects_all_regions_nn{nn_mi}.png",
        dpi=300,
        bbox_inches="tight"
    )
    plt.close()

    pivot.to_csv(
        f"{results_path}/heatmap_movie_effects_all_regions_nn{nn_mi}.csv"
)
    
def plot_movie_heatmap_sensitive_regions_ordered(coef_df, res_df, results_path, nn_mi, roi_names):
    """
    Heatmap: rows = movie-sensitive regions (ordered by roi_names),
             cols = movies,
             values = mean_effect
    """
    if coef_df.empty or res_df.empty:
        return

    # Select movie-sensitive regions
    sensitive_regions = res_df.loc[
        res_df["movie_sensitive"] == True, "region"
    ].tolist()

    if len(sensitive_regions) == 0:
        return

    # Keep only those that are also in roi_names (preserve order!)
    ordered_regions = [r for r in roi_names if r in sensitive_regions]

    if len(ordered_regions) == 0:
        return

    # Filter coefficient dataframe
    heat_df = coef_df[coef_df["region"].isin(ordered_regions)].copy()

    # Pivot
    pivot = heat_df.pivot(index="region", columns="movie", values="mean_effect")

    # Enforce order
    pivot = pivot.reindex(ordered_regions)

    # Plot
    plt.figure(figsize=(1.2 * max(4, pivot.shape[1]), 0.35 * max(8, pivot.shape[0])))
    plt.imshow(pivot.values, aspect="auto")
    plt.colorbar(label="Predicted mean effect")
    plt.xticks(range(len(pivot.columns)), pivot.columns, rotation=45, ha="right")
    plt.yticks(range(len(pivot.index)), pivot.index)
    plt.title("Movie effects across movie-sensitive regions (ROI order)")
    plt.tight_layout()

    plt.savefig(
        f"{results_path}/heatmap_movie_effects_movie_sensitive_regions_ordered_nn{nn_mi}.png",
        dpi=300,
        bbox_inches="tight"
    )
    plt.close()

    # Save values
    pivot.to_csv(
        f"{results_path}/heatmap_movie_effects_movie_sensitive_regions_ordered_nn{nn_mi}.csv"
    )    

def main(base_path, proj, nn_mi, atlas_path, roi_names):

    results_path = f"{base_path}/results_run_sTOPF_v2_data_{proj}/results_nn{nn_mi}"

    results_out_path = f"{results_path}/movie_sensitivity_nn{nn_mi}"
    os.makedirs(results_out_path, exist_ok=True)
    results_glass_brains_out_path = f"{results_path}/movie_sensitivity_nn{nn_mi}/glass_brains"
    os.makedirs(results_glass_brains_out_path, exist_ok=True)
    results_data_out_path = f"{results_path}/movie_sensitivity_nn{nn_mi}/tables"
    os.makedirs(results_data_out_path, exist_ok=True)

    res_path = f"{results_out_path}/movie_specificity_nn{nn_mi}.csv"
    coef_path = f"{results_out_path}/movie_specificity_per_movie_nn{nn_mi}.csv"

    res_df = pd.read_csv(res_path)
    coef_df = pd.read_csv(coef_path)

    make_brain_maps(
        res_df=res_df,
        coef_df=coef_df,
        results_g_path=results_glass_brains_out_path,
        results_t_path=results_data_out_path,
        nn_mi=nn_mi,
        atlas_path=atlas_path,
        roi_names=roi_names,
    )

    plot_top_sensitive_regions(res_df, results_data_out_path, nn_mi)
    plot_region_movie_profiles(coef_df, res_df, results_data_out_path, nn_mi)
    plot_movie_heatmap(coef_df, res_df, results_data_out_path, nn_mi)
    plot_movie_heatmap_all_regions(coef_df, results_data_out_path, nn_mi, roi_names)
    plot_movie_heatmap_sensitive_regions_ordered(coef_df, res_df, results_data_out_path, nn_mi, roi_names)
   

# Execute script
if __name__ == "__main__":
    main()
