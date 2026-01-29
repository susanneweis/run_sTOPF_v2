from pathlib import Path
import pandas as pd

# --------------------
# configuration
# --------------------
movies = ["DD", "DMW", "DPS", "FG", "LIB", "S", "SS", "TGTBTU"]
emotions = ["ANGST", "EKEL", "FREUDE", "SURPRISE", "TRAUER", "WUT"]

in_dir = Path("path/to/input_csvs")     # <-- change
out_dir = Path("path/to/output_csvs")   # <-- change
out_dir.mkdir(parents=True, exist_ok=True)

# --------------------
# processing
# --------------------
for movie in movies:
    mean_tcs = {}
    expected_T = None

    for emotion in emotions:
        fp = in_dir / f"{movie}_{emotion}.csv"
        if not fp.exists():
            raise FileNotFoundError(fp)

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

    out_file = out_dir / f"{movie}_mean_timecourses.csv"
    out_df.to_csv(out_file, index=False)

    print(f"✅ wrote {out_file} | timepoints={out_df.shape[0]} | emotions={out_df.shape[1]}")

# Execute script
if __name__ == "__main__":
    main()