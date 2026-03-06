
import pandas as pd
import numpy as np
import os
import sys
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import socket

import _results_1_summarize_classifications
import _results_2_assess_stability
import _results_3_assess_stability_corr
import _results_4_sex_type_distributions
import _results_5_CV_classification_results


base_path =  "/Users/sweis/Data/Arbeit/Juseless/data/project/brainvar_sexdiff_movies" 

#results = "results_run_sTOPF_v2"
#nn_values = [3,5,10,15,20,25,30,35,40,50,60,70,80,90,100]

results = "results_run_sTOPF_v2_data_v4"
#nn_values = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]
nn_values = [11]

#_results_1_summarize_classifications.main(base_path,results,nn_values)
#_results_2_assess_stability.main(base_path,results,nn_values)
#_results_3_assess_stability_corr.main(base_path,results)
#_results_4_sex_type_distributions.main(base_path,results)

for nn in nn_values:        
    path = f"{base_path}/{results}/results_nn{nn}/ind_classification_CV/"
    file = "classification_CV_corr"
    col = "cv_balanced_accuracy_mean"
    _results_5_classification_results.main(path,file,col)
    path = f"{base_path}/{results}/results_nn{nn}/ind_classification_CV/"
    file = "classification_CV_corr"
    col = "test_balanced_accuracy"
    _results_5_CV_classification_results.main(path,file,col)