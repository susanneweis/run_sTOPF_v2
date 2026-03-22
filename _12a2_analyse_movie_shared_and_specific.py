import os
import numpy as np
import pandas as pd
from _util_glass_brains_borders import create_glassbrains


def main(base_path, proj, nn_mi, mov_prop, max_regions, atlas_path, roi_names):

    """
    Computes:
    1) shared map across all movies (mean per region)
    2) leave-one-movie-out mean map for each movie
    3) movie-specific residual map: movie - mean(other movies)
    4) normalized residual map: (movie - mean(other movies)) / sd(other movies)
    5) ranked tables for most shared / most variable / most movie-specific regions
    """
    movies = list(mov_prop.keys())
    # actual movies
    movies = movies[:-3]   # excluding last two, as in your script

    results_path = f"{base_path}/results_run_sTOPF_v2_data_{proj}/results_nn{nn_mi}"
    in_path = f"{results_path}/compare_time_courses_nn{nn_mi}"

    results_out_path = f"{results_path}/movie_sensitivity_b_nn{nn_mi}"
    os.makedirs(results_out_path, exist_ok=True)

    results_g_path = f"{results_path}/movie_sensitivity_b_nn{nn_mi}/glass_brains"
    os.makedirs(results_g_path, exist_ok=True)

    metric="corr"
    region_col="region"
    eps=1e-8
    top_n=max_regions

    # ------------------------------------------------------------------
    # 1. Read all movie files and collect the chosen metric
    # ------------------------------------------------------------------
    dfs = []

    for movie in movies:

        file_path = f"{in_path}/results_compare_time_courses_{movie}.csv"

        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found for movie {movie}: {file_path}")

        df = pd.read_csv(file_path)

        if region_col not in df.columns:
            raise ValueError(f"Column '{region_col}' not found in {file_path}")

        if metric not in df.columns:
            raise ValueError(f"Column '{metric}' not found in {file_path}")

        curr = df[[region_col, metric]].copy()
        curr = curr.rename(columns={metric: movie})
        dfs.append(curr)

    # ------------------------------------------------------------------
    # 2. Merge into one region x movie matrix
    # ------------------------------------------------------------------
    data_wide = dfs[0]
    for curr in dfs[1:]:
        data_wide = data_wide.merge(curr, on=region_col, how="outer")

    data_wide = data_wide.sort_values(region_col).reset_index(drop=True)
    movie_cols = list(movies)

    data_wide.to_csv(
        os.path.join(results_out_path, f"region_by_movie_{metric}.csv"),
        index=False
    )

    # ------------------------------------------------------------------
    # 3. Shared map across all movies
    # ------------------------------------------------------------------
    shared_map = data_wide[[region_col]].copy()
    shared_map["mean_all_movies"] = data_wide[movie_cols].mean(axis=1)
    shared_map["std_all_movies"] = data_wide[movie_cols].std(axis=1, ddof=1)
    shared_map["var_all_movies"] = data_wide[movie_cols].var(axis=1, ddof=1)
    shared_map["abs_mean_all_movies"] = shared_map["mean_all_movies"].abs()

    # strong and stable
    shared_map["stability_score"] = (
        shared_map["abs_mean_all_movies"] / (shared_map["std_all_movies"] + eps)
    )

    shared_map_file = f"{results_out_path}/shared_map_all_movies_{metric}.csv"
    shared_map.to_csv(shared_map_file, index=False)

    create_glassbrains(
        shared_map_file,
        "stability_score",
        "region",
        roi_names,
        atlas_path,
        f"All movies: stability score {metric}",
        results_g_path,
        f"Stability_all_movies_{metric}",
        "continuous", 
        shared_map["stability_score"].min(), 
        shared_map["stability_score"].max()
    )


    # ------------------------------------------------------------------
    # 4. Ranked tables across all movies
    # ------------------------------------------------------------------
    # Most shared = high absolute mean + low variability
    most_shared = shared_map.sort_values(
        by="stability_score", ascending=False
    ).reset_index(drop=True)

    most_shared.to_csv(
        os.path.join(results_out_path, f"ranked_most_shared_regions_{metric}.csv"),
        index=False
    )

    # most_shared.head(top_n).to_csv(
    #     os.path.join(results_out_path, f"top_{top_n}_most_shared_regions_{metric}.csv"),
    #     index=False
    # )

    # Most variable across movies
    most_variable = shared_map.sort_values(
        by="std_all_movies", ascending=False
    ).reset_index(drop=True)

    most_variable.to_csv(
        os.path.join(results_out_path, f"ranked_most_variable_regions_{metric}.csv"),
        index=False
    )

    # most_variable.head(top_n).to_csv(
    #     os.path.join(results_out_path, f"top_{top_n}_most_variable_regions_{metric}.csv"),
    #     index=False
    # )

    # Strongest mean effects regardless of stability
    strongest_mean = shared_map.sort_values(
        by="abs_mean_all_movies", ascending=False
    ).reset_index(drop=True)

    strongest_mean.to_csv(
        os.path.join(results_out_path, f"ranked_strongest_mean_regions_{metric}.csv"),
        index=False
    )

    # strongest_mean.head(top_n).to_csv(
    #     os.path.join(results_out_path, f"top_{top_n}_strongest_mean_regions_{metric}.csv"),
    #     index=False
    # )

    # ------------------------------------------------------------------
    # 5. Per-movie leave-one-out residuals and ranked outputs
    # ------------------------------------------------------------------
    summary_rows = []

    for movie in movie_cols:
        other_movies = [m for m in movie_cols if m != movie]

        out_df = data_wide[[region_col]].copy()
        out_df[f"{movie}_score"] = data_wide[movie]
        out_df["mean_other_movies"] = data_wide[other_movies].mean(axis=1)
        out_df["std_other_movies"] = data_wide[other_movies].std(axis=1, ddof=1)
        out_df["var_other_movies"] = data_wide[other_movies].var(axis=1, ddof=1)

        out_df["residual"] = out_df[f"{movie}_score"] - out_df["mean_other_movies"]
        out_df["normalized_residual"] = (
            out_df["residual"] / (out_df["std_other_movies"] + eps)
        )

        out_df["abs_residual"] = out_df["residual"].abs()
        out_df["abs_normalized_residual"] = out_df["normalized_residual"].abs()

        spec_out_file = f"{results_out_path}/movie_specific_{movie}_{metric}.csv"
        # Save full per-movie table
        out_df.to_csv(spec_out_file,index=False)

        create_glassbrains(
            spec_out_file,
            "normalized_residual",
            "region",
            roi_names,
            atlas_path,
            f"{movie} {metric}: Normalized residual",
            results_g_path,
            f"Normalized_residual_{movie}_{metric}",
            "continuous",
            -10,
            10
        )


        # -----------------------------
        # Ranked residual tables
        # -----------------------------
        # Positive residuals = stronger than expected in this movie
        pos_resid = out_df.sort_values(by="residual", ascending=False).reset_index(drop=True)
        pos_resid.to_csv(
            os.path.join(results_out_path, f"ranked_positive_residuals_{movie}_{metric}.csv"),
            index=False
        )
        # pos_resid.head(top_n).to_csv(
        #     os.path.join(results_out_path, f"top_{top_n}_positive_residuals_{movie}_{metric}.csv"),
        #     index=False
        # )

        # Negative residuals = weaker than expected in this movie
        neg_resid = out_df.sort_values(by="residual", ascending=True).reset_index(drop=True)
        neg_resid.to_csv(
            os.path.join(results_out_path, f"ranked_negative_residuals_{movie}_{metric}.csv"),
            index=False
        )
        # neg_resid.head(top_n).to_csv(
        #     os.path.join(results_out_path, f"top_{top_n}_negative_residuals_{movie}_{metric}.csv"),
        #     index=False
        # )

        # Absolute residuals = strongest deviations regardless of direction
        abs_resid = out_df.sort_values(by="abs_residual", ascending=False).reset_index(drop=True)
        abs_resid.to_csv(
            os.path.join(results_out_path, f"ranked_absolute_residuals_{movie}_{metric}.csv"),
            index=False
        )
        # abs_resid.head(top_n).to_csv(
        #     os.path.join(results_out_path, f"top_{top_n}_absolute_residuals_{movie}_{metric}.csv"),
        #     index=False
        # )

        # Positive normalized residuals
        pos_norm = out_df.sort_values(by="normalized_residual", ascending=False).reset_index(drop=True)
        pos_norm.to_csv(
            os.path.join(results_out_path, f"ranked_positive_normalized_residuals_{movie}_{metric}.csv"),
            index=False
        )
        # pos_norm.head(top_n).to_csv(
        #     os.path.join(results_out_path, f"top_{top_n}_positive_normalized_residuals_{movie}_{metric}.csv"),
        #     index=False
        # )

        # Negative normalized residuals
        neg_norm = out_df.sort_values(by="normalized_residual", ascending=True).reset_index(drop=True)
        neg_norm.to_csv(
            os.path.join(results_out_path, f"ranked_negative_normalized_residuals_{movie}_{metric}.csv"),
            index=False
        )
        # neg_norm.head(top_n).to_csv(
        #     os.path.join(results_out_path, f"top_{top_n}_negative_normalized_residuals_{movie}_{metric}.csv"),
        #     index=False
        # )

        # Absolute normalized residuals
        abs_norm = out_df.sort_values(by="abs_normalized_residual", ascending=False).reset_index(drop=True)
        abs_norm.to_csv(
            os.path.join(results_out_path, f"ranked_absolute_normalized_residuals_{movie}_{metric}.csv"),
            index=False
        )
        # abs_norm.head(top_n).to_csv(
        #     os.path.join(results_out_path, f"top_{top_n}_absolute_normalized_residuals_{movie}_{metric}.csv"),
        #     index=False
        # )

        # Movie-level summary
        summary_rows.append({
            "movie": movie,
            "mean_abs_residual": out_df["abs_residual"].mean(),
            "max_abs_residual": out_df["abs_residual"].max(),
            "mean_abs_normalized_residual": out_df["abs_normalized_residual"].mean(),
            "max_abs_normalized_residual": out_df["abs_normalized_residual"].max(),
        })

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(
        os.path.join(results_out_path, f"movie_specific_summary_{metric}.csv"),
        index=False
    )

    print("Done.")
    print(f"Saved outputs to: {results_out_path}")


# Execute script
if __name__ == "__main__":
    main()
