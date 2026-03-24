import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.cluster.hierarchy import linkage, leaves_list


def create_heat_df_from_movie_files(movie_specific_dir, use_abs):
    """
    Create region x movie matrix from movie-specific files.

    Returns
    -------
    heat_df : pd.DataFrame
        Rows = regions
        Columns = movies
        Values = normalized residuals
    """

    movie_specific_dir = Path(movie_specific_dir)
    movie_files = sorted(movie_specific_dir.glob("movie_specific_*_corr.csv"))
    movie_files = movie_files[:-1]

    if len(movie_files) == 0:
        raise ValueError("No movie-specific files found.")

    residual_col = "abs_normalized_residual" if use_abs else "normalized_residual"

    dfs = []
    for f in movie_files:
        movie_name = f.stem.replace("movie_specific_", "").replace("_corr", "")
        df = pd.read_csv(f)

        if not {"region", residual_col}.issubset(df.columns):
            raise ValueError(f"{f.name} missing required columns.")

        tmp = df[["region", residual_col]].copy()
        tmp["movie"] = movie_name
        dfs.append(tmp)

    all_df = pd.concat(dfs, ignore_index=True)

    heat_df = all_df.pivot(
        index="region",
        columns="movie",
        values=residual_col
    )

    return heat_df


def summarize_movie_residual_matrix(heat_df, out_path=None, prefix="movie_specific"):
    """
    Same as before: clustered heatmap + region summary + movie correlation
    """

    heat_df = heat_df.copy()

    # -----------------------------
    # clustering
    # -----------------------------
    cluster_input = heat_df.fillna(0)
    Z = linkage(cluster_input.values, method="ward")
    row_order = leaves_list(Z)
    clustered_df = heat_df.iloc[row_order]

    # -----------------------------
    # region summary
    # -----------------------------
    region_summary = pd.DataFrame(index=heat_df.index)
    region_summary["mean_residual"] = heat_df.mean(axis=1)
    region_summary["std_residual"] = heat_df.std(axis=1)
    region_summary["mean_abs_residual"] = heat_df.abs().mean(axis=1)
    region_summary["max_abs_residual"] = heat_df.abs().max(axis=1)

    region_summary = region_summary.sort_values("mean_abs_residual", ascending=False)

    # -----------------------------
    # movie correlation
    # -----------------------------
    movie_corr = heat_df.corr()

    # -----------------------------
    # plot 1: clustered heatmap
    # -----------------------------
    plt.figure(figsize=(1.2 * max(4, clustered_df.shape[1]),
                        0.35 * max(8, clustered_df.shape[0])))

    vmax = np.nanmax(np.abs(clustered_df.values))
    if vmax == 0:
        vmax = 1

    im = plt.imshow(clustered_df.values, aspect="auto", vmin=-vmax, vmax=vmax)
    plt.colorbar(im, label="Normalized residual")

    plt.xticks(range(len(clustered_df.columns)), clustered_df.columns, rotation=45, ha="right")
    plt.yticks(range(len(clustered_df.index)), clustered_df.index)

    plt.title("Clustered heatmap (movie-specific residuals)")
    plt.tight_layout()

    if out_path:
        plt.savefig(f"{out_path}/{prefix}_clustered_heatmap.png", dpi=300)
    plt.show()

    # -----------------------------
    # plot 2: region summary
    # -----------------------------
    plot_df = region_summary.reset_index().rename(columns={"index": "region"})

    plt.figure(figsize=(8, 0.35 * max(8, len(plot_df))))
    plt.barh(plot_df["region"].astype(str), plot_df["mean_abs_residual"])
    plt.gca().invert_yaxis()

    plt.xlabel("Mean abs normalized residual")
    plt.title("Region summary (movie-specificity)")
    plt.tight_layout()

    if out_path:
        plt.savefig(f"{out_path}/{prefix}_region_summary.png", dpi=300)
    plt.show()

    # -----------------------------
    # plot 3: movie correlation
    # -----------------------------
    plt.figure(figsize=(6, 5))
    im = plt.imshow(movie_corr.values, aspect="auto", vmin=-1, vmax=1)
    plt.colorbar(im, label="Correlation")

    plt.xticks(range(len(movie_corr.columns)), movie_corr.columns, rotation=45, ha="right")
    plt.yticks(range(len(movie_corr.index)), movie_corr.index)

    for i in range(movie_corr.shape[0]):
        for j in range(movie_corr.shape[1]):
            plt.text(j, i, f"{movie_corr.iloc[i, j]:.2f}",
                     ha="center", va="center", fontsize=8)

    plt.title("Movie correlation matrix")
    plt.tight_layout()

    if out_path:
        plt.savefig(f"{out_path}/{prefix}_movie_corr.png", dpi=300)
    plt.show()

    return clustered_df, region_summary, movie_corr


def main(base_path, proj, nn_mi, atlas_path, roi_names):
    
    top_n=30
    use_abs=False
    metric = "corr"

    """
    FULL WORKFLOW:
    1. create heat_df from movie files
    2. select least stable regions
    3. summarize + plot
    """
    results_path = f"{base_path}/results_run_sTOPF_v2_data_{proj}/results_nn{nn_mi}/movie_sensitivity_no_sub_fac_with_ss_nn{nn_mi}"
    #results_out_path = f"{results_path}/movie_sensitivity_no_sub_fac_nn{nn_mi}"
    #os.makedirs(results_out_path, exist_ok=True)
    #results_glass_brains_out_path = f"{results_path}/movie_sensitivity_nn{nn_mi}/glass_brains"
    #os.makedirs(results_glass_brains_out_path, exist_ok=True)
    #results_data_out_path = f"{results_path}/movie_sensitivity_nn{nn_mi}/tables"
    #os.makedirs(results_data_out_path, exist_ok=True)

    # -----------------------------
    # load shared map
    # -----------------------------
    shared_map_file = f"{results_path}/shared_map_all_movies_{metric}.csv"
    shared_df = pd.read_csv(shared_map_file)

    least_regions = (
        shared_df.sort_values("stability_score", ascending=True)
                 .head(top_n)["region"]
                 .tolist()
    )

    # -----------------------------
    # create matrix
    # -----------------------------
    heat_df = create_heat_df_from_movie_files(results_path,use_abs)

    # restrict to least stable
    heat_df = heat_df.loc[least_regions]

    # -----------------------------
    # summarize
    # -----------------------------
    summarize_movie_residual_matrix(heat_df,out_path=results_path,prefix=f"least_stable_top{top_n}")