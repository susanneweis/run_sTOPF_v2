
import pandas as pd
import numpy as np
import os
import sys
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import socket

import _add_data_1_emotion_anno

base_path =  "/Users/sweis/Data/Arbeit/Juseless/data/project/brainvar_sexdiff_movies" 

results = "results_annotations"

_add_data_1_emotion_anno.main(base_path,results,nn_values)
