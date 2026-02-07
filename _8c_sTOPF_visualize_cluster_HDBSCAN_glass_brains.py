import pandas as pd
import numpy as np
import nibabel as nib
import os
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from nilearn.plotting import plot_glass_brain
from matplotlib import cm
from matplotlib import colors

# Assign unique numerical IDs to each region name and adds them as a new column
def assign_roi_ids(df):
    unique_regions = df['roi_name'].drop_duplicates().tolist()
    region_to_id = {region: i+1 for i, region in enumerate(unique_regions)}  
    df['ROI_ID'] = df['roi_name'].map(region_to_id)
    return df, region_to_id

def create_img_for_glassbrain_plot(stat_to_plot, atlas_path, n_roi):
    # Load atlas
    atlas_img = nib.load(atlas_path)
    atlas_data = atlas_img.get_fdata()
    
    # Create empty output image
    new_img = np.zeros(atlas_data.shape)
    
    # Sanity checks
    a_roi = int(np.max(atlas_data))
    if n_roi != a_roi:
        print(f"Mismatch between input ROIs ({n_roi}) and atlas ROIs ({a_roi})")

    # Check if stat_to_plot has the expected length
    if len(stat_to_plot) != n_roi:
        raise ValueError(f"Length of stat_to_plot ({len(stat_to_plot)}) does not match expected n_roi ({n_roi})")

    # Reshape data if needed
    stat_to_plot = np.reshape(stat_to_plot, (n_roi, 1))

    # Assign values to ROIs
    for roi in range(n_roi):
        voxel_indices = np.where(atlas_data == roi + 1)  # 1-based indexing
        if voxel_indices[0].size == 0:
            print(f"ROI {roi+1} not found in atlas.")
        new_img[voxel_indices] = stat_to_plot[roi]

    # Return Nifti image
    img_nii = nib.Nifti1Image(new_img, atlas_img.affine)
    return img_nii

def fill_glassbrain(n_r,res_df,column):
    # Initialize array for all ROIs
    roi_values = np.full(n_r, np.nan)

    # Fill in corr (convert Region to 0-based index)
    for _, row in res_df.iterrows():
        region_index = int(row['ROI_ID']) - 1  
        if 0 <= region_index < n_r:
            roi_values[region_index] = row[column]
    return roi_values

def create_glassbrains(vals, at_path, nrois, title_str,o_file,min_val,max_val):
    
    if min_val < max_val:
        # Create image
        img = create_img_for_glassbrain_plot(vals, at_path, nrois)

        # Define output filename

        cmap = cm.RdBu_r  # Diverging colormap with blue (negative) and red (positive)

        n_labels = int(max_val - min_val +1)

        # create a discrete colormap
        cmap = colors.ListedColormap(
           plt.cm.tab20(np.linspace(0, 1, n_labels))
        )
                
        # Plot and save glass brain
        #plot_glass_brain(img, threshold=0, vmax=max_val, vmin=min_val,display_mode='lyrz', colorbar=True, cmap = cmap, title=title_str, plot_abs=False)

        plot_glass_brain(
            img,
            cmap=cmap,
            vmin=min_val - 0.5,
            vmax=max_val + 0.5,
            colorbar=True,
            title=title_str,
            plot_abs=False
        )   

        plt.savefig(o_file, bbox_inches='tight',dpi=300)
        plt.close()
    
        print(f"Saved brain map: {o_file}")


def main(base_path,proj,nn_mi,movies_properties):

    results_path = f"{base_path}/results_run_sTOPF_v2_data_{proj}/results_nn{nn_mi}/ind_classification_CV_clustered_HDBSCAN/clusters/"
    results_out_path = f"{base_path}/results_run_sTOPF_v2_data_{proj}/results_nn{nn_mi}/ind_classification_CV_clustered_HDBSCAN/clusters/glass_brains"
    os.makedirs(results_out_path, exist_ok=True)
    
    data_path = f"{base_path}/data_run_sTOPF_{proj}"
    atlas_path = f"{data_path}/Susanne_Schaefer_436.nii"
    
    movies = list(movies_properties.keys())

    for mv_str in movies:
        
        outpath = f"{results_out_path}/{mv_str}"
        os.makedirs(outpath, exist_ok=True)

        cluster_assign_file = f"{results_path}/{mv_str}/roi_cluster_labels_HDBSCAN_clusters_corr.csv"
        cluster_assign = pd.read_csv(cluster_assign_file)
        
        n_roi = cluster_assign["roi_name"].nunique()
        
        cluster_brain, region_to_id_f = assign_roi_ids(cluster_assign)
        
        roi_values = fill_glassbrain(n_roi,cluster_brain,"cluster")

        # Define output filename

        title = f"Clusters {mv_str} HDBSCAN Clusters corr"
        
        output_file = f"{outpath}/{mv_str}_HDBSCAN_clusters_corr.png"

        min_val = roi_values.min()
        max_val = roi_values.max()
        
        create_glassbrains(roi_values, atlas_path, n_roi, title,output_file,min_val,max_val)


        cluster_assign_file = f"{results_path}/{mv_str}/roi_cluster_labels_HDBSCAN_clusters_nn{nn_mi}.csv"
        cluster_assign = pd.read_csv(cluster_assign_file)
        
        n_roi = cluster_assign["roi_name"].nunique()
        
        cluster_brain, region_to_id_f = assign_roi_ids(cluster_assign)
        
        roi_values = fill_glassbrain(n_roi,cluster_brain,"cluster")

        # Define output filename

        title = f"Clusters {mv_str} HDBSCAN Clusters _nn{nn_mi}"
        
        output_file = f"{outpath}/{mv_str}_HDBSCAN_clusters_nn{nn_mi}.png"

        min_val = roi_values.min()
        max_val = roi_values.max()
        
        create_glassbrains(roi_values, atlas_path, n_roi, title,output_file,min_val,max_val)

# Execute script
if __name__ == "__main__":
    main()
