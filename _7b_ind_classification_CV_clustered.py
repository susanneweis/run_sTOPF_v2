import pandas as pd
import os
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.inspection import permutation_importance
from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.cluster import AgglomerativeClustering
import matplotlib.pyplot as plt
from matplotlib import colormaps
import umap


import numpy as np
import pandas as pd

# Julearn 
from julearn import run_cross_validation
from julearn.pipeline import PipelineCreator
from julearn import scoring
from julearn.model_selection import StratifiedBootstrap
from julearn.stats.corrected_ttest import corrected_ttest

# from julearn.viz import plot_scores

import numpy as np
import pandas as pd

def collapse_rois_to_clusters(df, roi_cols, roi_labels, id_col, sex_col, agg, prefix):
    """
    df: dataframe with id_col, sex_col, and roi_cols
    roi_cols: list of ROI column names (must be length n_rois)
    roi_labels: array of cluster labels per ROI, length n_rois, aligned with roi_cols
    """
    roi_labels = np.asarray(roi_labels)
    if len(roi_cols) != len(roi_labels):
        raise ValueError(f"roi_cols ({len(roi_cols)}) and roi_labels ({len(roi_labels)}) must have same length.")

    # keep metadata
    out = df[[id_col, sex_col]].copy()

    # ROI matrix in the exact order of roi_cols
    X = df[roi_cols].to_numpy()

    clusters = np.unique(roi_labels)
    for k in clusters:
        mask = roi_labels == k
        if agg == "mean":
            out[f"{prefix}_{int(k):02d}"] = X[:, mask].mean(axis=1)
        elif agg == "median":
            out[f"{prefix}_{int(k):02d}"] = np.median(X[:, mask], axis=1)
        else:
            raise ValueError("agg must be 'mean' or 'median'")

    return out

def save_clustering(roi_name, roi_lab, roi_corr, metric, K_cluster, out_path, movie):
    roi_df = pd.DataFrame({
        "roi_name": roi_name,
        "cluster": roi_lab
    })

    cluster_out_path = f"{out_path}/clusters/{movie}"
    if not os.path.exists(cluster_out_path):
        os.makedirs(cluster_out_path, exist_ok=True) # Create the output directory if it doesn't exist


    # optional but useful
    roi_df["is_noise"] = roi_df["cluster"] == -1
    
    roi_df.to_csv(f"{cluster_out_path}/roi_cluster_labels_{K_cluster}_clusters_{metric}.csv", index=False)

    roi_corr_df = pd.DataFrame(
        roi_corr,
        index=roi_name,
        columns=roi_name
    )
    
    roi_corr_df.to_csv(f"{cluster_out_path}/roi_cluster_correlation_{K_cluster}_clusters_{metric}.csv", index=False)

    cluster_summary = (
        roi_df
        .groupby("cluster")
        .size()
        .reset_index(name="n_rois")
    )

    cluster_summary.to_csv(f"{cluster_out_path}/cluster_summary_{K_cluster}_clusters_{metric}.csv", index=False)

    order = np.argsort(roi_lab)
    roi_corr_sorted = roi_corr[order][:, order]
    roi_names_sorted = np.array(roi_name)[order]

    roi_corr_sorted_df = pd.DataFrame(
        roi_corr_sorted,
        index=roi_names_sorted,
        columns=roi_names_sorted
    )

    roi_corr_sorted_df.to_csv(f"{cluster_out_path}/roi_cluster_correlation_sorted_{K_cluster}_clusters_{metric}.csv",  index=False)


def plot_clusters(D, out_file, roi_labels):
    # 1) 2D embedding for visualization
    um2 = umap.UMAP(
        metric="precomputed",
        n_neighbors=50,
        min_dist=0.0,
        n_components=2,
        random_state=0
    )
    Z2 = um2.fit_transform(D)

    # 2) Colors from HDBSCAN clustering (done in 10D)
    labels = roi_labels
    is_noise = labels == -1
    clusters = np.unique(labels[~is_noise])
    n_clusters = len(clusters)

  
    cmap = colormaps["tab20"]

    color_map = {
        c: cmap(i / max(n_clusters - 1, 1))
        for i, c in enumerate(clusters)
    }

    colors = np.array([
        color_map.get(l, (0.6, 0.6, 0.6, 0.7))  # grey for noise
        for l in labels
]   )

    # 3) Plot
    plt.figure(figsize=(6, 5))
    plt.scatter(Z2[:, 0], Z2[:, 1], c=colors, s=25, linewidths=0)

    plt.title("UMAP (2D) colored by HDBSCAN clusters (10D)")
    plt.xlabel("UMAP-1")
    plt.ylabel("UMAP-2")

    # Optional legend
    if n_clusters <= 20:
        for c in clusters:
            plt.scatter([], [], c=[color_map[c]], label=f"cluster {c}", s=40)
        if np.any(is_noise):
            plt.scatter([], [], c=[(0.6, 0.6, 0.6, 0.7)], label="noise (-1)", s=40)
        plt.legend(frameon=False, bbox_to_anchor=(1.02, 1), loc="upper left")

    plt.tight_layout()

    # 4) Save (IMPORTANT: before plt.show())
    plt.savefig(out_file, dpi=300, bbox_inches="tight")
    #plt.show()

    print(f"Saved figure to: {out_file}")

def main(base_path, proj, nn_mi,movies_properties, K_clust):
    results_path = f"{base_path}/results_run_sTOPF_v2_data_{proj}/results_nn{nn_mi}"

    results_out_path = f"{results_path}/ind_classification_CV_clustered"
    if not os.path.exists(results_out_path):
        os.makedirs(results_out_path, exist_ok=True) # Create the output directory if it doesn't exist

    results_out_path_fi = f"{results_out_path}/feature_importance"
    if not os.path.exists(results_out_path_fi):
        os.makedirs(results_out_path_fi, exist_ok=True) # Create the output directory if it doesn't exist

    ind_ex_path = f"{results_path}/individual_expression_all_nn{nn_mi}.csv"
    ind_ex_data = pd.read_csv(ind_ex_path)
    # subs = ind_ex_data["subject"].unique().tolist()

    sex_mapping = {1: 'male', 2: 'female'}

    #movies = ["dd", "s", "dps", "fg", "dmw", "lib", "tgtbtu", "ss", "rest_run-1", "rest_run-2"]
    movies = list(movies_properties.keys())

    #mutual information very low for rest - therefore for now only real movies
    #for now

    #movies_short = movies[:-2]

    all_feature_importances = []   # will hold tidy rows across all runs

    # Classification per Movie
    # change this if neccessary 
    
    out_csv_mi = f"{results_out_path}/classification_CV_nn{nn_mi}_clustered_{K_clust}clust.csv" 
    out_csv_corr = f"{results_out_path}/classification_CV_corr_clustered_{K_clust}clust.csv" 
    
    for curr_mov in movies:

        cluster_out_path = f"{results_out_path}/clusters/{curr_mov}"
        if not os.path.exists(cluster_out_path):
            os.makedirs(cluster_out_path, exist_ok=True) # Create the output directory if it doesn't exist


        curr_movie_data = ind_ex_data[ind_ex_data["movie"] == curr_mov]

        curr_data = curr_movie_data[["subject", "sex", "region", "fem_vs_mal_mi"]].copy()

        class_data = curr_data.pivot(index=["subject", "sex"], columns="region", values="fem_vs_mal_mi").reset_index()
        class_data.columns.name = None

        inv_sex_mapping = {v: k for k, v in sex_mapping.items()}

        # apply to DataFrame
        class_data["sex"] = class_data["sex"].map(inv_sex_mapping)

        meta_cols = ["subject", "sex"]
        roi_cols = [c for c in class_data.columns if c not in meta_cols]

        X = [c for c in class_data.columns if c in roi_cols]        
        y = "sex"

        # show data type of predictors
        X_types = {"continuous": X}

        class_data_train, class_data_test = train_test_split(
            class_data,
            test_size=0.1,
            random_state=42,
            stratify=class_data[y],   # important for classification
        )

        roi_data_train = class_data_train[roi_cols]
        X_roi_train = roi_data_train.to_numpy()

        roi_corr = np.corrcoef(X_roi_train.T)

        clustering = AgglomerativeClustering(
            n_clusters=K_clust,
            metric="precomputed",
            linkage="average"
        )

        roi_labels = clustering.fit_predict(1 - roi_corr)
        
        D = 1 - roi_corr 
        plotfile = f"{results_out_path}/clusters/{curr_mov}/roi_cluster_correlation_sorted_{K_clust}_clusters_nn{nn_mi}.png"
        plot_clusters(D, plotfile, roi_labels)

        # roi_cols must match the columns you used to build roi_corr / clustering
        # e.g. roi_cols = [c for c in class_data_train.columns if c not in ["participant", "sex"]]

        save_clustering(roi_cols, roi_labels, roi_corr, f"nn{nn_mi}", K_clust, results_out_path, curr_mov)

        train_cluster_data = collapse_rois_to_clusters(
            df=class_data_train,
            roi_cols=roi_cols,
            roi_labels=roi_labels,
            id_col="subject",
            sex_col="sex",
            agg="mean",
            prefix="cluster_"
        )

        test_cluster_data = collapse_rois_to_clusters(
            df=class_data_test,
            roi_cols=roi_cols,
            roi_labels=roi_labels,
            id_col="subject",
            sex_col="sex",
            agg="mean",
            prefix="cluster_"
        )

        meta_cols = ["subject", "sex"]
        roi_cols = [c for c in train_cluster_data.columns if c not in meta_cols]

        X = [c for c in train_cluster_data.columns if c in roi_cols]        
        y = "sex"
        X_types = {"continuous": X}

        cv = RepeatedKFold(n_splits=10, n_repeats=1, random_state=22)
        scoring = ["accuracy", "balanced_accuracy", "f1"]

        scores1, model1 = run_cross_validation(
            X=X,
            y=y,
            X_types=X_types, 
            data=train_cluster_data,
            model="svm",
            problem_type="classification",
            seed=200,
            return_estimator="final",
            return_train_score=True,
            #return_inspector=True,
            cv=cv,
            scoring=scoring,
        )

        # feature importance

        # X_test must be the same feature columns you used in X=...
        X_test = test_cluster_data[X]
        y_test = test_cluster_data[y]

        pi = permutation_importance(
            model1,
            X_test,
            y_test,
            scoring="balanced_accuracy",   # or "accuracy", "f1"
            n_repeats=50,
            random_state=0,
            n_jobs=-1,
        )

        feat_importance = pd.Series(pi.importances_mean, index=X).sort_values(ascending=False)
        print(feat_importance.head(20))

        # build tidy table: one row per feature (top-k)
        fi_df = feat_importance.reset_index()
        fi_df.columns = ["feature", "importance"]
        fi_df["rank"] = np.arange(1, len(fi_df) + 1)

        # add metadata so it’s meaningful later
        fi_df["movie"] = curr_mov                 # <-- must exist in your loop
        fi_df["metric"] = f"nn{nn_mi}"         # <-- set this in your loop (e.g., "mi", "corr")
        fi_df["cluster_num"] = K_clust # <-- your 10/20/...
        fi_df["model"] = "svm"
        fi_df["scoring"] = "balanced_accuracy"
        fi_df["n_repeats_perm"] = 50

        all_feature_importances.append(fi_df)

        y_true = test_cluster_data[y].to_numpy()
        y_pred = model1.predict(test_cluster_data[X])

        acc  = accuracy_score(y_true, y_pred)
        bacc = balanced_accuracy_score(y_true, y_pred)
        f1   = f1_score(y_true, y_pred)  

        # print(scores1)
        #print("Mean test accuracy:", scores1["test_accuracy"].mean())
        #print("Mean balanced test accuracy:", scores1["test_balanced_accuracy"].mean())
        #print("Mean false alarm rate:", scores1["test_f1"].mean())

        row = {
            "cluster_num": K_clust,
            "movie": curr_mov,
            "train_score_mean": float(scores1["train_accuracy"].mean()),
            "cv_accuracy_mean": float(scores1["test_accuracy"].mean()),
            "cv_balanced_accuracy_mean": float(scores1["test_balanced_accuracy"].mean()),
            "cv_f1_mean": float(scores1["test_f1"].mean()),
            "train_score_std": float(scores1["train_accuracy"].std()),
            "cv_accuracy_std": float(scores1["test_accuracy"].std()),
            "cv_balanced_accuracy_std": float(scores1["test_balanced_accuracy"].std()),
            "cv_f1_std": float(scores1["test_f1"].std()),
            "test_accuracy": acc,
            "test_balanced_accuracy": bacc,
            "test_f1": f1,
        }

        df_row = pd.DataFrame([row])

        # write header only if file doesn't exist yet (or is empty)
        out_path_mi = Path(out_csv_mi)
        write_header = (not out_path_mi.exists()) or (out_path_mi.stat().st_size == 0)
        df_row.to_csv(out_csv_mi, mode="a", header=write_header, index=False,float_format="%.2f")

    feature_importance_all = pd.concat(all_feature_importances, ignore_index=True)

    out_fp = f"{results_out_path_fi}/feature_importance_summary_{K_clust}cluster_nn{nn_mi}.csv"
    feature_importance_all.to_csv(out_fp, index=False)
    print(f"Saved feature importance table to: {out_fp}")


    all_feature_importances = []   # will hold tidy rows across all runs


    for curr_mov in movies:

        curr_movie_data = ind_ex_data[ind_ex_data["movie"] == curr_mov]

        curr_data = curr_movie_data[["subject", "sex", "region", "fem_vs_mal_corr"]].copy()

        class_data = curr_data.pivot(index=["subject", "sex"], columns="region", values="fem_vs_mal_corr").reset_index()
        class_data.columns.name = None

        inv_sex_mapping = {v: k for k, v in sex_mapping.items()}

        # apply to DataFrame
        class_data["sex"] = class_data["sex"].map(inv_sex_mapping)

        meta_cols = ["subject", "sex"]
        roi_cols = [c for c in class_data.columns if c not in meta_cols]

        X = [c for c in class_data.columns if c in roi_cols]  
        y = "sex"

        # show data type of predictors
        X_types = {"continuous": X}

        class_data_train, class_data_test = train_test_split(
            class_data,
            test_size=0.1,
            random_state=42,
            stratify=class_data[y],   # important for classification
        )

        roi_data_train = class_data_train[roi_cols]
        X_roi_train = roi_data_train.to_numpy()

        roi_corr = np.corrcoef(X_roi_train.T)

        clustering = AgglomerativeClustering(
            n_clusters=K_clust,
            metric="precomputed",
            linkage="average"
        )

        roi_labels = clustering.fit_predict(1 - roi_corr)

        D = 1 - roi_corr 
        plotfile = f"{results_out_path}/clusters/{curr_mov}/roi_cluster_correlation_sorted_{K_clust}_clusters_corr.png"
        plot_clusters(D, plotfile, roi_labels)


        # roi_cols must match the columns you used to build roi_corr / clustering
        # e.g. roi_cols = [c for c in class_data_train.columns if c not in ["participant", "sex"]]

        save_clustering(roi_cols, roi_labels, roi_corr, "corr", K_clust, results_out_path, curr_mov)

        train_cluster_data = collapse_rois_to_clusters(
            df=class_data_train,
            roi_cols=roi_cols,
            roi_labels=roi_labels,
            id_col="subject",
            sex_col="sex",
            agg="mean",
            prefix="cluster_"
        )

        test_cluster_data = collapse_rois_to_clusters(
            df=class_data_test,
            roi_cols=roi_cols,
            roi_labels=roi_labels,
            id_col="subject",
            sex_col="sex",
            agg="mean",
            prefix="cluster_"
        )

        meta_cols = ["subject", "sex"]
        roi_cols = [c for c in train_cluster_data.columns if c not in meta_cols]

        X = [c for c in train_cluster_data.columns if c in roi_cols]        
        y = "sex"
        X_types = {"continuous": X}


        cv = RepeatedKFold(n_splits=10, n_repeats=1, random_state=22)
        scoring = ["accuracy", "balanced_accuracy", "f1"]

        scores1, model1 = run_cross_validation(
            X=X,
            y=y,
            X_types=X_types, 
            data=train_cluster_data,
            model="svm",
            problem_type="classification",
            seed=200,
            return_estimator="final",
            return_train_score=True,
            #return_inspector=True,
            cv=cv,
            scoring=scoring,
        )

        # feature importance

        # X_test must be the same feature columns you used in X=...
        X_test = test_cluster_data[X]
        y_test = test_cluster_data[y]

        pi = permutation_importance(
            model1,
            X_test,
            y_test,
            scoring="balanced_accuracy",   # or "accuracy", "f1"
            n_repeats=50,
            random_state=0,
            n_jobs=-1,
        )

        feat_importance = pd.Series(pi.importances_mean, index=X).sort_values(ascending=False)
        print(feat_importance.head(20))

        # build tidy table: one row per feature (top-k)
        fi_df = feat_importance.reset_index()
        fi_df.columns = ["feature", "importance"]
        fi_df["rank"] = np.arange(1, len(fi_df) + 1)

        # add metadata so it’s meaningful later
        fi_df["movie"] = curr_mov                 # <-- must exist in your loop
        fi_df["metric"] = "corr"         # <-- set this in your loop (e.g., "mi", "corr")
        fi_df["cluster_num"] = K_clust # <-- your 10/20/...
        fi_df["model"] = "svm"
        fi_df["scoring"] = "balanced_accuracy"
        fi_df["n_repeats_perm"] = 50

        all_feature_importances.append(fi_df)

        y_true = test_cluster_data[y].to_numpy()
        y_pred = model1.predict(test_cluster_data[X])

        acc  = accuracy_score(y_true, y_pred)
        bacc = balanced_accuracy_score(y_true, y_pred)
        f1   = f1_score(y_true, y_pred)  
        
        #print(scores1)
        #print("Mean test accuracy:", scores1["test_accuracy"].mean())
        #print("Mean balanced test accuracy:", scores1["test_balanced_accuracy"].mean())
        #print("Mean false alarm rate:", scores1["test_f1"].mean())
    
        row = {
            "cluster_num": K_clust,
            "movie": curr_mov,
            "train_score_mean": float(scores1["train_accuracy"].mean()),
            "cv_accuracy_mean": float(scores1["test_accuracy"].mean()),
            "cv_balanced_accuracy_mean": float(scores1["test_balanced_accuracy"].mean()),
            "cv_f1_mean": float(scores1["test_f1"].mean()),
            "train_score_std": float(scores1["train_accuracy"].std()),
            "cv_accuracy_std": float(scores1["test_accuracy"].std()),
            "cv_balanced_accuracy_std": float(scores1["test_balanced_accuracy"].std()),
            "cv_f1_std": float(scores1["test_f1"].std()),
            "test_accuracy": acc,
            "test_balanced_accuracy": bacc,
            "test_f1": f1,
        }

        df_row = pd.DataFrame([row])

        # write header only if file doesn't exist yet (or is empty)
        out_path_corr = Path(out_csv_corr)
        write_header = (not out_path_corr.exists()) or (out_path_corr.stat().st_size == 0)
        df_row.to_csv(out_csv_corr, mode="a", header=write_header, index=False,float_format="%.2f")
    

    feature_importance_all = pd.concat(all_feature_importances, ignore_index=True)

    out_fp = f"{results_out_path_fi}/feature_importance_summary_{K_clust}cluster_corr.csv"
    feature_importance_all.to_csv(out_fp, index=False)
    print(f"Saved feature importance table to: {out_fp}")

# Execute script
if __name__ == "__main__":
    main()