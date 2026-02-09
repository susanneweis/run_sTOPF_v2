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

def main(base_path,proj,nn_mi,movies_properties,k_clust):

    results_path = f"{base_path}/results_run_sTOPF_v2_data_{proj}/results_nn{nn_mi}/ind_classification_CV_clustered/clusters/"
    results_out_path = f"{base_path}/results_run_sTOPF_v2_data_{proj}/results_nn{nn_mi}/ind_classification_CV_clustered/clusters/"
    os.makedirs(results_out_path, exist_ok=True)
    
    data_path = f"{base_path}/data_run_sTOPF_{proj}"
    atlas_path = f"{data_path}/Susanne_Schaefer_436.nii"
    
    movies = list(movies_properties.keys())

    for mv_str in movies:
        
        outpath = f"{results_out_path}/{mv_str}"
        os.makedirs(outpath, exist_ok=True)

        cluster_assign_file = f"{results_path}/{mv_str}/roi_cluster_labels_{k_clust}_clusters_corr.csv"
        roi_fill_name = "cluster"
        roi_name_file = f"{data_path}/ROI_names.csv"
        title = f"Clusters {mv_str} {k_clust} Clusters corr"
        name_str = f"{mv_str}_{k_clust}_clusters_corr.png"
        create_glassbrains(cluster_assign_file, roi_fill_name, roi_name_file, atlas_path, title, outpath, name_str)

        cluster_assign_file = f"{results_path}/{mv_str}/roi_cluster_labels_{k_clust}_clusters_nn{nn_mi}.csv"
        roi_fill_name = "cluster"
        roi_name_file = f"{data_path}/ROI_names.csv"
        title = f"Clusters {mv_str} {k_clust} Clusters nn{nn_mi}"
        name_str = f"{mv_str}_{k_clust}_clusters_nn{nn_mi}.png"
        create_glassbrains(cluster_assign_file, roi_fill_name, roi_name_file, atlas_path, title, outpath, name_str)
        
# Execute script
if __name__ == "__main__":
    main()
