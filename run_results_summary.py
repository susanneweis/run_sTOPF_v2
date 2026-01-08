
import pandas as pd
import numpy as np
import os
import sys
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import socket

import _results_1_summarize_classifications
# import _1b_sTOPF_loo_PCA

base_path =  "/Users/sweis/Data/Arbeit/Juseless/data/project/brainvar_sexdiff_movies" 
results = "results_run_sTOPF_v2_data_v2"
nn = nn_values = [1,2,3,4,5,6,7,8,9,10,15]

_results_1_summarize_classifications.main(base_path,results,nn)