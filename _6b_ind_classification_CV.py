import pandas as pd
import os
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

# Julearn 
from julearn import run_cross_validation
from julearn.pipeline import PipelineCreator
from julearn import scoring
from julearn.model_selection import StratifiedBootstrap
from julearn.stats.corrected_ttest import corrected_ttest
from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import RepeatedStratifiedKFold

def main(base_path, proj, nn_mi,movies_properties,quant):
    results_path = f"{base_path}/results_run_sTOPF_v2_data_{proj}/results_nn{nn_mi}"
    results_out_path = f"{base_path}/results_run_sTOPF_v2_data_{proj}/results_nn{nn_mi}/ind_classification_CV_nn{nn_mi}"

    if not os.path.exists(results_out_path):
        os.makedirs(results_out_path, exist_ok=True) # Create the output directory if it doesn't exist

    ind_ex_path = f"{results_path}/individual_expression_all_nn{nn_mi}.csv"
    ind_ex_data = pd.read_csv(ind_ex_path)
    # subs = ind_ex_data["subject"].unique().tolist()

    sex_mapping = {1: 'male', 2: 'female'}

    #quant = 10
    quantile = quant*0.01

    #movies = ["dd", "s", "dps", "fg", "dmw", "lib", "tgtbtu", "ss", "rest_run-1", "rest_run-2"]
    movies = list(movies_properties.keys())

    # Classification per Movie
    for curr_mov in movies:

        out_csv_mi = f"{results_path}/classification_CV_nn{nn_mi}.csv" 
        out_csv_corr = f"{results_path}/classification_CV_corr.csv" 

        # adjust variable naming and print out results
        
        cmp_tc_path = f"{results_path}/compare_time_courses_nn{nn_mi}/results_compare_time_courses_{curr_mov}.csv" 
        cmp_tc_data = pd.read_csv(cmp_tc_path)

        thresh = cmp_tc_data["mutual_inf"].quantile(quantile)

        diff_regs = cmp_tc_data.loc[cmp_tc_data["mutual_inf"] < thresh, "region"].tolist()
        curr_movie_data = ind_ex_data[ind_ex_data["movie"] == curr_mov]
        curr_mov_reg_data = curr_movie_data[curr_movie_data["region"].isin(diff_regs)]

        curr_data = curr_mov_reg_data[["subject", "sex", "region", "fem_vs_mal_mi"]].copy()

        class_data = curr_data.pivot(index=["subject", "sex"], columns="region", values="fem_vs_mal_mi").reset_index()
        class_data.columns.name = None

        n_region_cols = class_data.shape[1] - 2
        new_columns = ["subject", "sex"] + [f"R{i}" for i in range(1, n_region_cols + 1)]
        class_data.columns = new_columns

        inv_sex_mapping = {v: k for k, v in sex_mapping.items()}

        # apply to DataFrame
        class_data["sex"] = class_data["sex"].map(inv_sex_mapping)

        X = [c for c in class_data.columns if c not in ["subject", "sex"]]        
        y = "sex"

        # show data type of predictors
        X_types = {"continuous": X}

        class_data_train, class_data_test = train_test_split(
            class_data,
            test_size=0.1,
            random_state=42,
            stratify=class_data[y],   # important for classification
        )

        cv = RepeatedKFold(n_splits=10, n_repeats=1, random_state=22)
        scoring = ["accuracy", "balanced_accuracy", "f1"]

        scores1, model1 = run_cross_validation(
            X=X,
            y=y,
            X_types=X_types, 
            data=class_data_train,
            model="svm",
            problem_type="classification",
            seed=200,
            return_estimator="final",
            return_train_score=True,
            #return_inspector=True,
            cv=cv,
            scoring=scoring,
        )

        y_true = class_data_test[y].to_numpy()
        y_pred = model1.predict(class_data_test[X])

        acc  = accuracy_score(y_true, y_pred)
        bacc = balanced_accuracy_score(y_true, y_pred)
        f1   = f1_score(y_true, y_pred)  

        # print(scores1)
        #print("Mean test accuracy:", scores1["test_accuracy"].mean())
        #print("Mean balanced test accuracy:", scores1["test_balanced_accuracy"].mean())
        #print("Mean false alarm rate:", scores1["test_f1"].mean())

        row = {
            "top_reg": quant,
            "movie": curr_mov,
            "train_score": float(scores1["train_accuracy"].mean()),
            "cv_accuracy_mean": float(scores1["test_accuracy"].mean()),
            "cv_balanced_accuracy_mean": float(scores1["test_balanced_accuracy"].mean()),
            "cv_f1_mean": float(scores1["test_f1"].mean()),
            "test_accuracy": acc,
            "test_balanced_accuracy": bacc,
            "test_f1": f1,
        }

        df_row = pd.DataFrame([row])

        # write header only if file doesn't exist yet (or is empty)
        out_path_mi = Path(out_csv_mi)
        write_header = (not out_path_mi.exists()) or (out_path_mi.stat().st_size == 0)
        df_row.to_csv(out_csv_mi, mode="a", header=write_header, index=False,float_format="%.2f")



        thresh = cmp_tc_data["corr"].quantile(quantile)

        diff_regs = cmp_tc_data.loc[cmp_tc_data["corr"] < thresh, "region"].tolist()
        curr_movie_data = ind_ex_data[ind_ex_data["movie"] == curr_mov]
        curr_mov_reg_data = curr_movie_data[curr_movie_data["region"].isin(diff_regs)]

        curr_data = curr_mov_reg_data[["subject", "sex", "region", "fem_vs_mal_corr"]].copy()

        class_data = curr_data.pivot(index=["subject", "sex"], columns="region", values="fem_vs_mal_corr").reset_index()
        class_data.columns.name = None

        n_region_cols = class_data.shape[1] - 2
        new_columns = ["subject", "sex"] + [f"R{i}" for i in range(1, n_region_cols + 1)]
        class_data.columns = new_columns

        inv_sex_mapping = {v: k for k, v in sex_mapping.items()}

        # apply to DataFrame
        class_data["sex"] = class_data["sex"].map(inv_sex_mapping)

        X = [c for c in class_data.columns if c not in ["subject", "sex"]]        
        y = "sex"

        # show data type of predictors
        X_types = {"continuous": X}

        class_data_train, class_data_test = train_test_split(
            class_data,
            test_size=0.1,
            random_state=42,
            stratify=class_data[y],   # important for classification
        )

        cv = RepeatedKFold(n_splits=10, n_repeats=1, random_state=22)
        scoring = ["accuracy", "balanced_accuracy", "f1"]

        scores1, model1 = run_cross_validation(
            X=X,
            y=y,
            X_types=X_types, 
            data=class_data_train,
            model="svm",
            problem_type="classification",
            seed=200,
            return_estimator="final",
            return_train_score=True,
            #return_inspector=True,
            cv=cv,
            scoring=scoring,
        )

        y_true = class_data_test[y].to_numpy()
        y_pred = model1.predict(class_data_test[X])

        acc  = accuracy_score(y_true, y_pred)
        bacc = balanced_accuracy_score(y_true, y_pred)
        f1   = f1_score(y_true, y_pred)  
        
        #print(scores1)
        #print("Mean test accuracy:", scores1["test_accuracy"].mean())
        #print("Mean balanced test accuracy:", scores1["test_balanced_accuracy"].mean())
        #print("Mean false alarm rate:", scores1["test_f1"].mean())
    
        row = {
            "top_reg": quant,
            "movie": curr_mov,
            "train_score": float(scores1["train_accuracy"].mean()),
            "cv_accuracy_mean": float(scores1["test_accuracy"].mean()),
            "cv_balanced_accuracy_mean": float(scores1["test_balanced_accuracy"].mean()),
            "cv_f1_mean": float(scores1["test_f1"].mean()),
            "test_accuracy": acc,
            "test_balanced_accuracy": bacc,
            "test_f1": f1,
        }

        df_row = pd.DataFrame([row])

        # write header only if file doesn't exist yet (or is empty)
        out_path_corr = Path(out_csv_corr)
        write_header = (not out_path_corr.exists()) or (out_path_corr.stat().st_size == 0)
        df_row.to_csv(out_csv_corr, mode="a", header=write_header, index=False,float_format="%.2f")
    

# Execute script
if __name__ == "__main__":
    main()