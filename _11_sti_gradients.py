import numpy as np
import pandas as pd
import os
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from _util_glass_brains import create_glassbrains


def plot_variance(expl,movie,out,met): 
    grad = np.arange(1, len(expl) + 1)

    # cumulative variance
    cumulative = np.cumsum(expl)

    plt.figure(figsize=(7,4))

    # explained variance
    plt.plot(grad, expl, marker="o", label="Explained variance")

    # cumulative variance
    plt.plot(grad, cumulative, marker="s", label="Cumulative variance")

    plt.xlabel("Gradient")
    plt.ylabel("Variance explained")
    plt.title(f"PCA Varience eplained {movie} {met}")

    plt.xticks(grad)
    plt.ylim(0, 1)
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.savefig(f"{out}/{movie}_explained_variance_{met}.png", dpi=300)
    
    # optional: only keep if you really want to display interactively
    # plt.show()
    plt.close()

def main(base_path, proj, nn_mi, mov_prop):

    results_path = f"{base_path}/results_run_sTOPF_v2_data_{proj}/results_nn{nn_mi}"

    results_out_path = f"{results_path}/sti_gradients_nn{nn_mi}"
    if not os.path.exists(results_out_path):
        os.makedirs(results_out_path, exist_ok=True) # Create the output directory if it doesn't exist

    ind_ex_path = f"{results_path}/individual_expression_all_nn{nn_mi}.csv"
    ind_ex_data = pd.read_csv(ind_ex_path)
    
    #movies = ["dd", "s", "dps", "fg", "dmw", "lib", "tgtbtu", "ss", "rest_run-1", "rest_run-2"]
    movies = list(mov_prop.keys())
    movies = movies + ["concat"]

    for curr_mov in movies:
        
        for metric in ["corr", f"nn{nn_mi}"]:

            if metric == "corr":
                column = "fem_vs_mal_corr"
            else:
                column = "fem_vs_mal_mi"

            # filter for this movie
            df_movie = ind_ex_data[ind_ex_data["movie"] == curr_mov]

            # create subjects × regions matrix
            sim_df = df_movie.pivot(index="subject", columns="region", values=column)

            sim = sim_df.to_numpy()

            # correlation across subjects
            corr_data = np.corrcoef(sim, rowvar=False)

            # optional: NaNs entfernen
            corr_data = np.nan_to_num(corr_data)

            pca = PCA(n_components=10)
            gradients = pca.fit_transform(corr_data)

            explained = pca.explained_variance_ratio_
            # gradient numbers

            grad_df = pd.DataFrame(
                gradients,
                index=sim_df.columns,  # region names
                columns=[f"PC{i+1}" for i in range(gradients.shape[1])]
            )
            grad_df.to_csv(f"{results_out_path}/{curr_mov}_gradients_{metric}.csv")

            expl_df = pd.DataFrame({
                "PC": np.arange(1, len(explained)+1),
                "explained_variance": explained,
                "cumulative_variance": np.cumsum(explained)
            })

            expl_df.to_csv(f"{results_out_path}/{curr_mov}_explained_variance_{metric}.csv", index=False)

            plot_variance(explained,curr_mov,results_out_path,metric)

            data_path = f"{base_path}/data_run_sTOPF_{proj}"
            atlas_path = f"{data_path}/Susanne_Schaefer_436.nii"
            roi_name_file = f"{data_path}/ROI_names.csv"
            roi_names = pd.read_csv(roi_name_file)["roi_name"].tolist()
            grad_file = f"{results_out_path}/{curr_mov}_gradients_{metric}.csv"  
            roi_value_name = "region"
            
            for pc in range(1, 5):  # PC1–PC4

                roi_fill_name = f"PC{pc}"
                title = f"Gradient {pc} {curr_mov} {metric}"
                name_str = f"Grad{pc}_{curr_mov}_{metric}"

                create_glassbrains(
                    grad_file,
                    roi_fill_name,
                    roi_value_name,
                    roi_names,
                    atlas_path,
                    title,
                    results_out_path,
                    name_str,
                    "continuous"
                )
            

