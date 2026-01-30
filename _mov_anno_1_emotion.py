from pathlib import Path
import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np


def main(base_path,movies, results,TR):
    # --------------------
    # configuration
    # --------------------
    emotions = ["ANGST", "EKEL", "FREUDE", "SURPRISE", "TRAUER", "WUT"]

    in_path = f"{base_path}/movies_annotations/annotation_emotions/compiled_data"
    
    out_path = f"{base_path}/{results}"

    if not os.path.exists(out_path):
        os.makedirs(out_path, exist_ok=True) # Create the output directory if it doesn't exist

    # --------------------
    # processing
    # --------------------
    for movie in movies:
        mean_tcs = {}
        expected_T = None

        for emotion in emotions:
            fp = f"{in_path}/{movie}_{emotion}.csv"

            # no header:
            # col 0 = participant, cols 1..T = time points
            df = pd.read_csv(fp, header=None)

            if df.shape[1] < 2:
                raise ValueError(f"{fp} has <2 columns. Expected participant + timepoints.")

            values = df.iloc[:, 1:]  # drop participant column
            values = values.apply(pd.to_numeric, errors="coerce")  # robust conversion

            # check timepoints count matches across emotions for the same movie
            T = values.shape[1]
            if expected_T is None:
                expected_T = T
            elif T != expected_T:
                raise ValueError(
                    f"Timepoints mismatch for {movie}: expected {expected_T}, "
                   f"but {emotion} has {T} in {fp.name}"
               )

            # mean across participants per timepoint -> length T
            mean_tc = values.mean(axis=0, skipna=True)
            mean_tcs[emotion] = mean_tc.reset_index(drop=True)

        # rows = time points, columns = emotions
        out_df = pd.DataFrame(mean_tcs, columns=emotions)

        out_file = f"{out_path}/{movie}_mean_timecourses.csv"

        out_df.to_csv(out_file, index=False, float_format="%.3f")

        print(f"✅ wrote {out_file} | timepoints={out_df.shape[0]} | emotions={out_df.shape[1]}")

    plot_dir = out_path    # where PNGs should go

    for movie in movies:
        fp = f"{plot_dir}/{movie}_mean_timecourses.csv"

        df = pd.read_csv(fp)  # columns are emotions, rows are timepoints

        # ensure correct column order (and ignore extra columns if any)
        df = df[[c for c in emotions if c in df.columns]]

        n_tp = len(df)
        time_sec = np.arange(n_tp) * TR

        plt.figure(figsize=(12, 5))

        #for emo in df.columns:
        #    plt.plot(df[emo].values, label=emo)

        for emo in df.columns:
            plt.plot(time_sec, df[emo].values, label=emo)

        plt.title(f"{movie} – Mean emotion time courses")
        plt.xlabel("Time (s)")
        plt.ylabel("Mean value")
        plt.legend(ncol=3, fontsize=9)
        plt.tight_layout()

        out_png = f"{plot_dir}/{movie}_mean_timecourses.png"
        plt.savefig(out_png, dpi=200)
        plt.close()

        print(f"✅ saved {out_png}")

# Execute script
if __name__ == "__main__":
    main()