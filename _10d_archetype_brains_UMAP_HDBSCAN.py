import pandas as pd
import numpy as np


def main(base_path, proj, nn_mi,movies_properties):

    results_path = f"{base_path}/results_run_sTOPF_v2_data_{proj}/results_nn{nn_mi}"
    arch_path = f"{results_path}/archetype_clusters_UMAP_HDBSCAN"

    movies = list(movies_properties.keys())
 
    for curr_mov in movies:
        
        for metric in ["corr", f"nn{nn_mi}"]:

            if metric == "corr":
                column = "fem_vs_mal_corr"
            else:
                column = "fem_vs_mal_mi"
            
            df_clusters = pd.read_csv(f"{arch_path}/Archetypes_{curr_mov}_UMAP_HDBSCAN_{metric}.csv")
            df_expr = pd.read_csv(f"{results_path}/individual_expression_all_nn{nn_mi}.csv")

            # merge over subject
            df = df_expr.merge(df_clusters, on="subject")

            #exclude noise
            #df = df[df["cluster"] != -1]

            # aggregate

            summary = (
                df.groupby(["cluster", "region"])
                .agg(
                    mean_corr=(column, "mean"),
                    var_corr=(column, "var"),
                )
                .reset_index()
            )

            out_path = f"{arch_path}/Archetype_{curr_mov}_brain_UMAP_HDBSCAN_{metric}.csv"
            summary.to_csv(out_path, index=False)



# Execute script
if __name__ == "__main__":
    main()