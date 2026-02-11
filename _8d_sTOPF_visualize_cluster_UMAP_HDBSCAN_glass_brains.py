import pandas as pd
import numpy as np
import nibabel as nib
import os
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from nilearn.plotting import plot_glass_brain
from matplotlib import cm
from matplotlib import colors
from _util_glass_brains import create_glassbrains

def main(base_path,proj,nn_mi,movies_properties):

    results_path = f"{base_path}/results_run_sTOPF_v2_data_{proj}/results_nn{nn_mi}/ind_classification_CV_clustered_UMAP_HDBSCAN/clusters/"
    os.makedirs(results_path, exist_ok=True)
    
    data_path = f"{base_path}/data_run_sTOPF_{proj}"
    atlas_path = f"{data_path}/Susanne_Schaefer_436.nii"
    roi_name_file = f"{data_path}/ROI_names.csv"
    roi_names = pd.read_csv(roi_name_file)["roi_name"].tolist()
    
    movies = list(movies_properties.keys())

    for mv_str in movies:
        
        out_path = f"{results_path}/glassbrains"
        os.makedirs(out_path, exist_ok=True)

        roi_value_name = "roi_name"
        roi_fill_name = "cluster"

        cluster_assign_file = f"{results_path}/{mv_str}/roi_cluster_labels_UMAP_HDBSCAN_clusters_corr.csv"
        title = f"Sex Score Clusters UMAP HDBSCAN {mv_str} corr"
        name_str = f"{mv_str}_UMAP_HDBSCAN_clusters_corr"
        create_glassbrains(cluster_assign_file, roi_fill_name, roi_value_name, roi_names, atlas_path, title, out_path, name_str,"discrete")

        cluster_assign_file = f"{results_path}/{mv_str}/roi_cluster_labels_UMAP_HDBSCAN_clusters_nn{nn_mi}.csv"
        title = f"Sex Score Clusters UMAP HDBSCAN {mv_str} nn{nn_mi}"
        name_str = f"{mv_str}_UMAP_HDBSCAN_clusters_nn{nn_mi}"
        create_glassbrains(cluster_assign_file, roi_fill_name, roi_value_name, roi_names, atlas_path, title, out_path, name_str,"discrete")


# Execute script
if __name__ == "__main__":
    main()
