import pandas as pd
import numpy as np

import os
from sklearn.cluster import HDBSCAN
import matplotlib.pyplot as plt
from matplotlib import colormaps
import umap

def save_archetypes(sub, arch_lab, sex, metric, out_path, movie):
    arch_df = pd.DataFrame({
        "subject": sub,
        "cluster": arch_lab,
        "sex": sex
    })

    #arch_out_path = f"{out_path}/{movie}"
    #os.makedirs(arch_out_path, exist_ok=True) # Create the output directory if it doesn't exist

    # optional but useful
    arch_df["is_noise"] = arch_df["cluster"] == -1
    
    arch_df.to_csv(f"{out_path}/Archetypes_{movie}_UMAP_HDBSCAN_{metric}.csv", index=False)

    
    # cluster_summary = (
    #     arch_df
    #     .groupby("cluster")
    #     .size()
    #     .reset_index(name="n subjects")
    # )

    cluster_summary = (
        arch_df
        .groupby(["cluster", "sex"])
        .size()
        .unstack(fill_value=0)
        .assign(n_subjects=lambda df: df.sum(axis=1))
        .reset_index()
    )

    cluster_summary.to_csv(f"{out_path}/Archetype_{movie}_summary_UMAP_HDBSCAN_{metric}.csv", index=False)

def plot_clusters(arch_labels, Z2, sex_list, res_path, movie, metric):

    out_file = f"{res_path}/{movie}_sex_sim_diff_archetype_clusters_UMAP_HDBSCAN_{metric}.png"

    # # --- checks ---
    labels = np.asarray(arch_labels)
    sex = np.asarray(sex_list)
    if labels.shape[0] != sex.shape[0]:
        raise ValueError(f"roi_labels (len={labels.shape[0]}) and sex_list (len={sex.shape[0]}) must match.")

    # 2) Colors for clusters
    is_noise = labels == -1
    clusters = np.unique(labels[~is_noise])
    n_clusters = len(clusters)

    cmap = colormaps["tab20"]
    color_map = {c: cmap(i / max(n_clusters - 1, 1)) for i, c in enumerate(clusters)}

    colors = np.array([
        color_map.get(l, (0.6, 0.6, 0.6, 0.7))  # grey for noise
        for l in labels
    ])

    # 3) Marker map for sex (customize as you like)
    # works for values like "F"/"M", "female"/"male", 0/1, etc.
    uniq_sex = np.unique(sex)
    marker_cycle = ["o", "^", "s", "D", "P", "X"]  # in case you have >2 groups
    marker_map = {sx: marker_cycle[i % len(marker_cycle)] for i, sx in enumerate(uniq_sex)}

    # 4) Plot: same colors, separate scatter per sex for different markers
    plt.figure(figsize=(6, 5))
    for sx in uniq_sex:
        mask = sex == sx
        plt.scatter(
            Z2[mask, 0], Z2[mask, 1],
            c=colors[mask],
            s=25,
            marker=marker_map[sx],
            linewidths=0,
            label=f"sex {sx}"
        )

    plt.title(f"Archetypes {metric} movie {movie}: color=cluster, marker=sex")
    plt.xlabel("UMAP-1")
    plt.ylabel("UMAP-2")

    # Legend: sex (markers)
    leg1 = plt.legend(frameon=False, title="Sex", loc="upper left", bbox_to_anchor=(1.02, 1))
    plt.gca().add_artist(leg1)

    # Optional legend: clusters (colors)
    if n_clusters <= 20:
        handles = []
        for c in clusters:
            h = plt.Line2D([0], [0], marker='o', linestyle='',
                           markerfacecolor=color_map[c], markeredgecolor='none',
                           markersize=7, label=f"cluster {c}")
            handles.append(h)
        if np.any(is_noise):
            handles.append(plt.Line2D([0], [0], marker='o', linestyle='',
                                      markerfacecolor=(0.6, 0.6, 0.6, 0.7),
                                      markeredgecolor='none',
                                      markersize=7, label="noise (-1)"))
        plt.legend(handles=handles, frameon=False, title="Cluster",
                   loc="lower left", bbox_to_anchor=(1.02, 0))

    plt.tight_layout()
    plt.savefig(out_file, dpi=300, bbox_inches="tight")
    print(f"Saved figure to: {out_file}")



def comp_umap (X_sim, umap_n_neigh, umap_min_dist, umap_n_comp, hbdscan_min_clust, hbdscan_min_samp):

    roi_corr = np.corrcoef(X_sim)
    D = 1 - roi_corr 

    um = umap.UMAP(
        metric="precomputed",
        n_neighbors=umap_n_neigh,    # try 15–50
        min_dist=umap_min_dist,
        #n_components=10,   # NOT 2 for clustering
        n_components = umap_n_comp,
        random_state=0
    )

    Z = um.fit_transform(D)   # D = 1 - abs(roi_corr)

    clusterer = HDBSCAN(
        min_cluster_size=hbdscan_min_clust,
        min_samples=hbdscan_min_samp
    )

    arch_labels = clusterer.fit_predict(Z) 
    return arch_labels, Z


def main(base_path, proj, nn_mi,movies_properties):
    results_path = f"{base_path}/results_run_sTOPF_v2_data_{proj}/results_nn{nn_mi}"

    results_out_path = f"{results_path}/archetype_clusters_UMAP_HDBSCAN"
    os.makedirs(results_out_path, exist_ok=True) # Create the output directory if it doesn't exist

    ind_ex_path = f"{results_path}/individual_expression_all_nn{nn_mi}.csv"
    ind_ex_data = pd.read_csv(ind_ex_path)

    sex_mapping = {1: 'male', 2: 'female'}

    movies = list(movies_properties.keys())
 
    for curr_mov in movies:

        curr_movie_data = ind_ex_data[ind_ex_data["movie"] == curr_mov]
        
        meta_cols = ["subject", "sex"]
        subs = curr_movie_data["subject"].tolist()
        sex = curr_movie_data["sex"].tolist()

        for metric in ["corr", f"nn{nn_mi}"]:

            if metric == "corr":
                column = "fem_vs_mal_corr"
            else:
                column = "fem_vs_mal_mi"

            curr_data = curr_movie_data[["subject", "sex", "region", column]].copy()
            cluster_data = curr_data.pivot(index=["subject", "sex"], columns="region", values=column).reset_index()
            cluster_data.columns.name = None
            subs = cluster_data["subject"].tolist()
            sex = cluster_data["sex"].tolist()

            roi_cols = [c for c in cluster_data.columns if c not in meta_cols]

            sim_data = cluster_data[roi_cols]
            X_sim = sim_data.to_numpy()

            arch_labels, Z = comp_umap (X_sim, 50, 0.0, 2, 10, 2)

            plot_clusters(arch_labels, Z, sex, results_out_path, curr_mov, metric)

            save_archetypes(subs, arch_labels, sex, metric, results_out_path, curr_mov)

# Execute script
if __name__ == "__main__":
    main()