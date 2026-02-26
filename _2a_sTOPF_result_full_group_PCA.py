import numpy as np
import pandas as pd
from pathlib import Path
import os
from scipy.stats import ttest_1samp
from scipy.stats import pearsonr
from sklearn.feature_selection import mutual_info_regression


def compare_timecourses(inpa, outfi, nn):
    # CSVs
    csv_f = "PC1_scores_female_allROI.csv"
    csv_m = "PC1_scores_male_allROI.csv" 
        
    female_csv = f"{inpa}/{csv_f}"
    male_csv = f"{inpa}/{csv_m}"

    region_col = "Region"
    value_col = "PC_score_1"

    fem_data = pd.read_csv(female_csv)
    fem_data = fem_data[[region_col, value_col]].copy()

    mal_data = pd.read_csv(male_csv)
    mal_data = mal_data[[region_col, value_col]].copy()

    zwi_df = pd.read_csv(female_csv)

    regions = zwi_df["Region"].drop_duplicates()
    #regions = sorted(set(fem.keys()).intersection(mal.keys()))
        
    print(f"Found {len(regions)} regions.")

    rows = []
    length_warnings = []

    for reg in regions:
        y_f = fem_data.loc[fem_data["Region"] == reg, "PC_score_1"]
        y_m = mal_data.loc[mal_data["Region"] == reg, "PC_score_1"]

        if y_f.size != y_m.size:
            length_warnings.append((r, y_f.size, y_m.size))

        d_abs = np.abs(y_f - y_m)
        t_stat, p_val = ttest_1samp(d_abs, 0, nan_policy='omit')
        ttest_sig = p_val <= 0.05
            
        # very rough correction
        p_val_corr = p_val * 436
        ttest_sig_corr = p_val_corr <= 0.05
        r, p_r = pearsonr(y_f, y_m)
        corr_sig = p_r <= 0.05

        df_yf = pd.DataFrame(y_f)
        df_ym = pd.DataFrame(y_m)

        df_yf = (df_yf - df_yf.mean()) / df_yf.std(ddof=1)
        df_ym = (df_ym - df_ym.mean()) / df_ym.std(ddof=1)

        mi_1 = mutual_info_regression(X=df_yf, y=df_ym, n_neighbors=nn, random_state=42)
        mi_2 = mutual_info_regression(X=df_ym, y=df_yf, n_neighbors=nn, random_state=42)

        mi = (mi_1+ mi_2)/2

        rows.append(dict(
            region=reg,
            n_samples=int(len(y_f)),
            p_val=p_val,
            t_stat=t_stat,  
            t_sig = ttest_sig,
            p_val_corr = p_val_corr,
            t_sig_corr = ttest_sig_corr,
            corr = r, 
            corr_p = p_r,  
            corr_sig = corr_sig,
            mutual_inf = mi[0]
        ))

        out = pd.DataFrame(rows)

    outpa = Path(outfi)
    out.to_csv(outpa, index=False)
    print(f"Saved: {outpa.resolve()}")

    if length_warnings:
        print("⚠️ Length mismatches (female vs male) — truncated to min length for testing:")
        for r, lf, lm in length_warnings[:10]:
            print(f"   {r}: female={lf}, male={lm}")
        if len(length_warnings) > 10:
            print(f"   ...and {len(length_warnings)-10} more regions")



def main(base_path,proj,nn_mi,movies_properties):
        
    results_path = f"{base_path}/results_run_sTOPF_v2_data_{proj}"
    results_out_path = f"{base_path}/results_run_sTOPF_v2_data_{proj}/results_nn{nn_mi}"

    movies = list(movies_properties.keys())

    #movies = ["dd", "s", "dps", "fg", "dmw", "lib", "tgtbtu", "ss", "rest_run-1", "rest_run-2"]

    for curr_mov in movies:
        in_path = f"{results_path}/results_PCA_all/{curr_mov}" 

        outpath = f"{results_out_path}/compare_time_courses_nn{nn_mi}"
        os.makedirs(outpath, exist_ok=True)
        out_csv = f"/{outpath}/results_compare_time_courses_{curr_mov}.csv"

        compare_timecourses(in_path, out_csv, nn_mi)

    in_path = f"{results_path}/results_PCA_all/concatenated_PCA" 
    
    outpath = f"{results_out_path}/compare_time_courses_nn{nn_mi}"
    out_csv = f"/{outpath}/results_compare_time_courses_concatenated.csv"
    compare_timecourses(in_path, out_csv, nn_mi)
    
# Execute script
if __name__ == "__main__":
    main()