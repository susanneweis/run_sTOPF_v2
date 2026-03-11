import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


def stab_curves(nn_values,avg,title,metric,res_path):

    plt.figure(figsize=(7, 5))
    plt.plot(nn_values, avg, marker="o")

    plt.xticks(nn_values)

    plt.xlabel("NN")
    plt.ylabel("Average Stability (r)")
    plt.title(f"Parameter Stability Curve - {metric}")

    plt.tight_layout()
    plt.savefig(f"{res_path}/stability_curve_{metric}_{title}.png", dpi=300)
    #plt.savefig(f"{res_path}/stability_curve_{metric}_{title}.pdf")
    plt.show()

    # -------------------------------------------------------
    # SAVE STABILITY CURVE
    # -------------------------------------------------------

    curve_df = pd.DataFrame({
        "nn": nn_values,
        "avg_stability_r": avg
    })

    curve_df.to_csv(
        f"{res_path}/stability_curve_{metric}_{title}.csv",
        index=False
    )



def main(base_path,res_path,nn_values):

    # -------------------------------------------------------
    # SETTINGS
    # -------------------------------------------------------
    results_path = f"{base_path}/{res_path}"

    metric = "mi"
    col = "fem_vs_mal_mi"

    # -------------------------------------------------------
    # LOAD VECTORS
    # -------------------------------------------------------

    vectors = {}

    for nn in nn_values:

        file = f"individual_expression_all_nn{nn}.csv"
        path = f"{results_path}/results_nn{nn}/{file}"

        df = pd.read_csv(path)
        df = df[~df["movie"].isin(["REST1", "REST2", "concat"])]
        df = df.sort_values(["subject", "movie", "region"])

        vectors[nn] = df[col].values

    # -------------------------------------------------------
    # COMPUTE STABILITY MATRIX
    # -------------------------------------------------------

    mat = np.zeros((len(nn_values), len(nn_values)))

    for i, nn1 in enumerate(nn_values):
        for j, nn2 in enumerate(nn_values):

            v1 = vectors[nn1]
            v2 = vectors[nn2]

            r = np.corrcoef(v1, v2)[0, 1]
            mat[i, j] = r


    # -------------------------------------------------------
    # HEATMAP
    # -------------------------------------------------------

    plt.figure(figsize=(9, 8))

    plt.imshow(mat, cmap="viridis", vmin=0.8, vmax=1)
    plt.colorbar(label="Continuous Stability (Pearson r)")
    plt.xticks(range(len(nn_values)), nn_values)
    plt.yticks(range(len(nn_values)), nn_values)

    # ---- add numbers in each cell ----
    for i in range(len(nn_values)):
        for j in range(len(nn_values)):
            plt.text(
                j, i,
                f"{mat[i, j]:.2f}",
                ha="center",
                va="center",
                color="white" if mat[i, j] < 0.9 else "black",
                fontsize=8
            )

    plt.title(f"Continuous Stability (Pearson r) - {metric}")
    plt.xlabel("NN")
    plt.ylabel("NN")

    plt.tight_layout()
    plt.savefig(f"{results_path}/stability_heatmap_{metric}.png", dpi=300)
    #plt.savefig(f"{results_path}/stability_heatmap_{metric}.pdf")
    plt.show()

    # -------------------------------------------------------
    # SAVE HEATMAP MATRIX
    # -------------------------------------------------------

    heatmap_df = pd.DataFrame(mat, index=nn_values, columns=nn_values)
    heatmap_df.to_csv(f"{results_path}/stability_heatmap_{metric}.csv")

    # -------------------------------------------------------
    # STABILITY CURVE
    # -------------------------------------------------------

    avg = []
    for i in range(len(nn_values)):
        others = np.delete(mat[i], i)
        avg.append(np.mean(others))

    title = "all_nn"    
    stab_curves(nn_values,avg,title,metric,results_path)

    avg = []
    for i in range(len(nn_values)):
        neigh = []
        if i - 1 >= 0:
            neigh.append(mat[i, i - 1])
        if i + 1 < len(nn_values):
            neigh.append(mat[i, i + 1])
        avg.append(np.mean(neigh))

    title = "1_neigh"    
    stab_curves(nn_values,avg,title,metric,results_path)

    avg = []
    for i in range(len(nn_values)):
        neigh = []
        for j in [i - 2, i - 1, i + 1, i + 2]:
            if 0 <= j < len(nn_values):
                neigh.append(mat[i, j])
        avg.append(np.mean(neigh))

    title = "2_neigh"    
    stab_curves(nn_values,avg,title,metric,results_path)


# Execute script
if __name__ == "__main__":
    main()
