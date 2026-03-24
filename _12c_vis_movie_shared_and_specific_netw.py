import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def add_schaefer_network(df, region_col="region"):
    """
    Add Schaefer-17 network label from ROI name.
    Assumes ROI names like LH_DefaultA_PFC_1.
    """
    out = df.copy()
    out["region"] = out["region"].apply(
        lambda x: x if str(x).startswith("17Networks_") else f"no_network_subcort_{x}"
    )
    out["network"] = out[region_col].astype(str).str.split("_").str[2]
    return out


def compute_network_summaries(res_df, coef_df):
    """
    Computes:
    1) proportion of movie-sensitive regions per network
    2) mean effect per movie within each network,
       restricted to movie-sensitive regions only
    """

    # add network labels
    res_df = add_schaefer_network(res_df, region_col="region")
    coef_df = add_schaefer_network(coef_df, region_col="region")

    # fixed Schaefer-17 order
    network_order = [
        "VisCent", "VisPeri",
        "SomMotA", "SomMotB",
        "DorsAttnA", "DorsAttnB",
        "SalVentAttnA", "SalVentAttnB",
        "LimbicA", "LimbicB",
        "ContA", "ContB", "ContC",
        "DefaultA", "DefaultB", "DefaultC",
        "TempPar", "subcort"
    ]

    # --------------------------------------------------
    # 1. Proportion sensitive per network
    # --------------------------------------------------
    net_sens = (
        res_df.groupby("network", observed=False)["movie_sensitive"]
        .mean()
        .reset_index(name="prop_sensitive")
    )

    net_sens["network"] = pd.Categorical(
        net_sens["network"],
        categories=network_order,
        ordered=True
    )
    net_sens = net_sens.sort_values("network").reset_index(drop=True)

    # --------------------------------------------------
    # 2. Mean effect per movie, only sensitive regions
    # --------------------------------------------------
    sensitive_regions = res_df.loc[res_df["movie_sensitive"] == True, "region"].unique()

    coef_sens = coef_df[coef_df["region"].isin(sensitive_regions)].copy()

    net_effects = (
        coef_sens.groupby(["network", "movie"], observed=False)["mean_effect"]
        .mean()
        .reset_index()
    )

    net_effects["network"] = pd.Categorical(
        net_effects["network"],
        categories=network_order,
        ordered=True
    )
    net_effects = net_effects.sort_values(["network", "movie"]).reset_index(drop=True)

    return net_sens, net_effects


def plot_prop_sensitive_per_network(net_sens, results_path, nn_mi):
    """
    Horizontal bar plot: proportion sensitive per network
    """
    plot_df = net_sens.dropna(subset=["network"]).copy()

    plt.figure(figsize=(8, 6))
    plt.barh(plot_df["network"].astype(str), plot_df["prop_sensitive"])
    plt.xlabel("Proportion of movie-sensitive regions")
    plt.ylabel("Network")
    plt.title("Movie sensitivity prevalence across Schaefer-17 networks")
    plt.xlim(0, 1)
    plt.tight_layout()

    out_png = os.path.join(
        results_path,
        f"network_prop_sensitive_nn{nn_mi}.png"
    )
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close()

    return out_png


def plot_network_effect_heatmap(net_effects, results_path, nn_mi):
    """
    Heatmap: rows = networks, cols = movies, values = mean effect
    using only movie-sensitive regions
    """
    if net_effects.empty:
        return None

    pivot = net_effects.pivot(index="network", columns="movie", values="mean_effect")

    plt.figure(figsize=(1.2 * max(4, pivot.shape[1]), 0.45 * max(6, pivot.shape[0])))
    im = plt.imshow(pivot.values, aspect="auto")
    plt.colorbar(im, label="Mean effect (sensitive regions only)")
    plt.xticks(range(len(pivot.columns)), pivot.columns, rotation=45, ha="right")
    plt.yticks(range(len(pivot.index)), pivot.index.astype(str))
    plt.xlabel("Movie")
    plt.ylabel("Network")
    plt.title("Movie effects across Schaefer-17 networks")
    plt.tight_layout()

    out_png = os.path.join(
        results_path,
        f"network_effect_heatmap_sensitive_only_nn{nn_mi}.png"
    )
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close()

    return out_png


def save_network_summary_tables(net_sens, net_effects, results_path, nn_mi):
    """
    Save summary tables as CSV
    """
    out_csv_1 = os.path.join(
        results_path,
        f"network_prop_sensitive_nn{nn_mi}.csv"
    )
    out_csv_2 = os.path.join(
        results_path,
        f"network_effects_sensitive_only_nn{nn_mi}.csv"
    )

    net_sens.to_csv(out_csv_1, index=False)
    net_effects.to_csv(out_csv_2, index=False)

    return out_csv_1, out_csv_2


def main(base_path, proj, nn_mi):
    """
    Full pipeline:
    - compute summaries
    - save csv
    - make plots
    """
    results_path = f"{base_path}/results_run_sTOPF_v2_data_{proj}/results_nn{nn_mi}"

    results_out_path = f"{results_path}/movie_sensitivity_nn{nn_mi}"
    os.makedirs(results_out_path, exist_ok=True)
    results_glass_brains_out_path = f"{results_path}/movie_sensitivity_nn{nn_mi}/glass_brains"
    os.makedirs(results_glass_brains_out_path, exist_ok=True)
    results_data_out_path = f"{results_path}/movie_sensitivity_nn{nn_mi}/tables"
    os.makedirs(results_data_out_path, exist_ok=True)
    os.makedirs(results_path, exist_ok=True)

    res_path = f"{results_out_path}/movie_specificity_nn{nn_mi}.csv"
    coef_path = f"{results_out_path}/movie_specificity_per_movie_nn{nn_mi}.csv"

    res_df = pd.read_csv(res_path)
    coef_df = pd.read_csv(coef_path)

    net_sens, net_effects = compute_network_summaries(res_df, coef_df)

    save_network_summary_tables(net_sens, net_effects, results_data_out_path, nn_mi)
    plot_prop_sensitive_per_network(net_sens, results_data_out_path, nn_mi)
    plot_network_effect_heatmap(net_effects, results_data_out_path, nn_mi)

# Execute script
if __name__ == "__main__":
    main()
