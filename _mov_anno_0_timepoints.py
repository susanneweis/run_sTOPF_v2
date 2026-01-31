from pathlib import Path
import pandas as pd

movies = ["DD", "DMV", "DPS", "FG", "LIB", "S", "SS", "TGTBTU", "REST1","REST2"]

base_dir1 = Path("/Users/sweis/Data/Arbeit/Juseless/data/project/brainvar_sexdiff_movies/data_run_sTOPF_v2/fMRIdata")
out_file = base_dir1 / "timepoint_min_max_per_movie_2026.csv"

rows = []

for movie in movies:
    fp = base_dir1 / f"BOLD_Schaefer_436_2025_mean_aggregation_task-{movie}_MOVIES.tsv"

    df = pd.read_csv(fp, sep="\t")

    # falls nötig: absichern, dass es numerisch ist
    df["timepoint"] = pd.to_numeric(df["timepoint"], errors="coerce")

    rows.append({
        "movie": movie,
        "timepoint_min": df["timepoint"].min(),
        "timepoint_max": df["timepoint"].max()
    })

out_df = pd.DataFrame(rows)
out_df.to_csv(out_file, index=False)

print(f"✅ Ergebnisse gespeichert in: {out_file}")


movies = ["dd", "dmw", "dps", "fg", "lib", "s", "ss", "tgtbtu","rest_run-1","rest_run-2"]

base_dir2 = Path("/Users/sweis/Data/Arbeit/Juseless/data/project/brainvar_hormone_movie/data/")
out_file = base_dir1 / "timepoint_min_max_per_movie_old.csv"

rows = []

for movie in movies:
    fp = base_dir2 / f"BOLD_Schaefer400_subcor36_mean_task-{movie}_MOVIES_INM7.csv"

    df = pd.read_csv(fp)

    # falls nötig: absichern, dass es numerisch ist
    df["timepoint"] = pd.to_numeric(df["timepoint"], errors="coerce")

    rows.append({
        "movie": movie,
        "timepoint_min": df["timepoint"].min(),
        "timepoint_max": df["timepoint"].max()
    })

out_df = pd.DataFrame(rows)
out_df.to_csv(out_file, index=False)

print(f"✅ Ergebnisse gespeichert in: {out_file}")


