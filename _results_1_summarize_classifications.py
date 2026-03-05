import os
import pandas as pd


def main(base_path,res_path,nn_values):
    # -------------------------
    # Configuration
    # -------------------------

    #nn_values = [1,2,3,4,5,6,7,8,9,10,15]
   
    perc_values = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]

    value_column = "overall classification correct"

    # -------------------------
    # Collect results
    # -------------------------
    results = []

    for nn in nn_values:
        row = {"nn": nn}

        for perc in perc_values:

            results_path = f"{base_path}/{res_path}/results_nn{nn}/ind_classification_nn{nn}/classification_subjects_across_actual_movies_nn{nn}_top_{perc}perc.csv"

            if not os.path.exists(results_path):
                row[perc] = None
                print(f"⚠️ Missing: {results_path}")
                continue

            df = pd.read_csv(results_path)

            # Ensure numeric
            df[value_column] = pd.to_numeric(df[value_column], errors="coerce")

            row[perc] = df[value_column].mean()

        results.append(row)

    # -------------------------
    # Create final DataFrame
    # -------------------------
    results_df = pd.DataFrame(results)
    results_df = results_df.set_index("nn").sort_index()

    # -------------------------
    # Save output
    # -------------------------
    out_file = f"{base_path}/{res_path}/mean_classification_across_nn_and_perc.csv"

    results_df.to_csv(out_file)

    # -------------------------
    # Collect results
    # -------------------------
    results_corr = []

    for perc in perc_values:

        results_path = f"{base_path}/{res_path}/results_nn{nn}/ind_classification_nn{nn}/classification_subjects_across_actual_movies_corr_top_{perc}perc.csv"

        if not os.path.exists(results_path):
            row[perc] = None
            print(f"⚠️ Missing: {results_path}")
            continue

        df = pd.read_csv(results_path)

        # Ensure numeric
        df[value_column] = pd.to_numeric(df[value_column], errors="coerce")

        results_corr.append(df[value_column].mean())

    # -------------------------
    # Create final DataFrame
    # -------------------------
    results_corr_df = pd.DataFrame(results_corr)

    # -------------------------
    # Save output
    # -------------------------
    out_file_corr = f"{base_path}/{res_path}/mean_classification_corr_across_perc.csv"

    results_corr_df.to_csv(out_file_corr)

    
    value_column = "classification correct"

    records = []

    for nn in nn_values:
        for perc in perc_values:
            fpath = (
                f"{base_path}/{res_path}/results_nn{nn}/ind_classification_nn{nn}/"
                f"classification_subjects_movies_nn{nn}_top_{perc}perc.csv"
            )

            if not os.path.exists(fpath):
                print(f"⚠️ Missing: {fpath}")
                continue

            df = pd.read_csv(fpath)
            df[value_column] = pd.to_numeric(df[value_column], errors="coerce")

            movie_means = (
                df.groupby("movie", as_index=False)[value_column]
                .mean()
                .rename(columns={value_column: f"mean_{value_column}"})
            )
            movie_means["nn"] = nn
            movie_means["perc"] = perc

            records.append(movie_means)

    results_df = pd.concat(records, ignore_index=True)
    results_df = results_df[["nn", "perc", "movie", f"mean_{value_column}"]].sort_values(["nn", "perc", "movie"])

    out_file = f"{base_path}/{res_path}/mean_{value_column}_by_movie_nn_perc_LONG.csv"
    results_df.to_csv(out_file, index=False)

    records = []

    for nn in nn_values:
        for perc in perc_values:
            fpath = (
                f"{base_path}/{res_path}/results_nn{nn}/ind_classification_nn{nn}/"
                f"classification_subjects_movies_corr_top_{perc}perc.csv"
            )

            if not os.path.exists(fpath):
                print(f"⚠️ Missing: {fpath}")
                continue

            df = pd.read_csv(fpath)
            df[value_column] = pd.to_numeric(df[value_column], errors="coerce")

            movie_means = (
                df.groupby("movie", as_index=False)[value_column]
                .mean()
                .rename(columns={value_column: f"mean_{value_column}"})
            )
            movie_means["nn"] = nn
            movie_means["perc"] = perc

            records.append(movie_means)

    results_df = pd.concat(records, ignore_index=True)
    results_df = results_df[["nn", "perc", "movie", f"mean_{value_column}"]].sort_values(["nn", "perc", "movie"])

    out_file = f"{base_path}/{res_path}/mean_{value_column}_by_movie_corr_perc_LONG.csv"
    results_df.to_csv(out_file, index=False)



# Execute script
if __name__ == "__main__":
    main()
