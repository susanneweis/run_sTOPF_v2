
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


base_path =  "/Users/sweis/Data/Arbeit/Juseless/data/project/brainvar_sexdiff_movies" 

#results = "results_run_sTOPF_v2"
#nn_values = [3,5,10,15,20,25,30,35,40,50,60,70,80,90,100]

results = "results_run_sTOPF_v2_data_v4"
nn_values = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]

#_results_1_summarize_classifications.main(base_path,results,nn_values)
#_results_2_assess_stability.main(base_path,results,nn_values)
#_results_3_assess_stability_corr.main(base_path,results)
_results_4_sex_type_distributions.main(base_path,results)