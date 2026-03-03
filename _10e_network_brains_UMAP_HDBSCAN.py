import pandas as pd
import numpy as np
from _util_glass_brains import create_glassbrains
import os 

def main(base_path, proj, nn_mi,movies_properties):

    results_path = f"{base_path}/results_run_sTOPF_v2_data_{proj}/results_nn{nn_mi}"

    netw_path = f"{results_path}/network_clusters_UMAP_HDBSCAN"
    netw_movie_path = f"{netw_path}/perMovie"
    netw_movie_glass_path = f"{netw_movie_path}/glassbrains"
    #netw_movie_glass_quant_path = f"{netw_movie_glass_path}/quantiles"
    netw_clust_movie_path = f"{netw_path}/perMovie/perCluster"
    netw_clust_movie_glass_path = f"{netw_path}/perMovie/perCluster/glassbrains" 

    os.makedirs(netw_movie_path, exist_ok=True) # Create the output directory if it doesn't exist
    os.makedirs(netw_movie_glass_path, exist_ok=True) # Create the output directory if it doesn't exist
    #os.makedirs(netw_movie_glass_quant_path, exist_ok=True) # Create the output directory if it doesn't exist
    os.makedirs(netw_clust_movie_path, exist_ok=True)
    os.makedirs(netw_clust_movie_glass_path, exist_ok=True)

    movies = list(movies_properties.keys())
    movies = movies + ["concat"]
 
    for curr_mov in movies:
        
        for metric in ["corr", f"nn{nn_mi}"]:

            if metric == "corr":
                column = "fem_vs_mal_corr"
            else:
                column = "fem_vs_mal_mi"
            
            df_clusters = pd.read_csv(f"{netw_path}/Networks_{curr_mov}_UMAP_HDBSCAN_{metric}.csv")
            df_expr = pd.read_csv(f"{results_path}/individual_expression_all_nn{nn_mi}.csv")

            # merge over subject
            df = df_expr.merge(df_clusters, on="region")

            #exclude noise
            #df = df[df["cluster"] != -1]

            # aggregate
            # is this correct? 

            summary = (
                df.groupby(["cluster", "region"])
                .agg(
                    mean = (column, "mean"),
                    var = (column, "var"),
                )
                .reset_index()
            )

            out_path = f"{netw_movie_path}/Networks_{curr_mov}_brain_UMAP_HDBSCAN_{metric}.csv"
            summary.to_csv(out_path, index=False)

            all_clust_data = pd.read_csv(out_path)
            # clusters = sorted(all_clust_data["cluster"].dropna().unique())

            # # sep files per cluster and glass brains    
            # for c in clusters:
            #     df_c = all_clust_data[all_clust_data["cluster"] == c].copy()

            #     q25 = df_c["mean"].quantile(0.25)
            #     q75 = df_c["mean"].quantile(0.75)
            #     q10 = df_c["mean"].quantile(0.10)
            #     q90 = df_c["mean"].quantile(0.90)
            #     q5 = df_c["mean"].quantile(0.05)
            #     q95 = df_c["mean"].quantile(0.95)


            #     # Add quantile flag column
            #     df_c["Quantile"] = np.select(
            #         [
            #             df_c["mean"] <= q5,
            #             df_c["mean"] >= q95
            #         ],
            #         [-1, 1],
            #         default=0
            #     )

            #     out_sep = f"{netw_movie_path}/Networks_{curr_mov}_brain_UMAP_HDBSCAN_{metric}_cluster-{c}.csv"
            #     df_c.to_csv(out_sep, index=False)

            data_path = f"{base_path}/data_run_sTOPF_{proj}"
            atlas_path = f"{data_path}/Susanne_Schaefer_436.nii"
            roi_name_file = f"{data_path}/ROI_names.csv"
            roi_names = pd.read_csv(roi_name_file)["roi_name"].tolist()

            #     roi_fill_name = "mean"
            #     roi_value_name = "region"
            #     title = f"Brain Networsk UMAP HDBSCAN Cluster {c} {curr_mov} {metric}"
            #     name_str = f"Brain_Network_{curr_mov}_Cluster_{c}_{metric}_UMAP_HDBSCAN"
                
            #     create_glassbrains(out_sep, roi_fill_name, roi_value_name, roi_names, atlas_path, title, netw_movie_glass_path, name_str,"continuous")
                
            #     # Quantile brains

            roi_fill_name = "cluster"
            roi_value_name = "region"
            title = f"Brain Networks UMAP HDBSCAN {curr_mov} {metric}"
            name_str = f"Brain_Networks_{curr_mov}_{metric}_UMAP_HDBSCAN"

            create_glassbrains(out_path, roi_fill_name, roi_value_name, roi_names, atlas_path, title, netw_movie_glass_path, name_str,"discrete")

            # create files and glass brains for each cluster


            in_file = out_path
            all_cluster_data = pd.read_csv(in_file)

            # clusters to export: all except -1 (include 0)
            clusters = sorted([c for c in all_cluster_data["cluster"].dropna().unique() if c != -1])

            for c in clusters:
                out = all_cluster_data.copy()

                mask = out["cluster"] == c
                out.loc[~mask, ["mean", "var"]] = 0  # keep region + cluster, zero out stats elsewhere

                # safe filename (handles float-ish cluster labels like 0.0)
                c_str = str(int(c)) if float(c).is_integer() else str(c).replace(".", "p")
                out_file = f"{netw_clust_movie_path}/Networks_{curr_mov}_brain_UMAP_HDBSCAN_{metric}_cluster_{c_str}.csv"
                out.to_csv(out_file, index=False)

                roi_fill_name = "mean"
                roi_value_name = "region"
                title = f"Brain Networks UMAP HDBSCAN {curr_mov} {metric} cluster {c}"
                name_str = f"Brain_Networks_{curr_mov}_{metric}_cluster_{c}_UMAP_HDBSCAN"

                create_glassbrains(out_file, roi_fill_name, roi_value_name, roi_names, atlas_path, title, netw_clust_movie_glass_path, name_str,"continuous")


# Execute script
if __name__ == "__main__":
    main()