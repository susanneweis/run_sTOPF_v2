import pandas as pd

def main(base_path,res_path):

    # Load data
    ind_exp_file = f"{base_path}/{res_path}/results_nn11/individual_expression_all_nn11.csv"
    ind_exp = pd.read_csv(ind_exp_file)

    # Helper function for range
    def value_range(x):
        return x.max() - x.min()

    # Group by movie and brain region
    summary = (
        ind_exp
        .groupby(["movie", "region"])
        .agg(
            # fem_vs_mal_corr
            corr_mean=("fem_vs_mal_corr", "mean"),
            corr_std=("fem_vs_mal_corr", "std"),
            corr_range=("fem_vs_mal_corr", value_range),
            corr_min=("fem_vs_mal_corr", "min"),
            corr_max=("fem_vs_mal_corr", "max"),

            # fem_vs_mal_mi
            mi_mean=("fem_vs_mal_mi", "mean"),
            mi_std=("fem_vs_mal_mi", "std"),
            mi_range=("fem_vs_mal_mi", value_range),
            mi_min=("fem_vs_mal_mi", "min"),
            mi_max=("fem_vs_mal_mi", "max"),
        )
        .reset_index()
    )

    # Save to CSV
    out_file = f"{base_path}/{res_path}/results_nn11/individual_expression_nn11_summary.csv"
    summary.to_csv(out_file, index=False)

    print("Saved to movie_region_summary_statistics.csv")

# Execute script
if __name__ == "__main__":
    main()
