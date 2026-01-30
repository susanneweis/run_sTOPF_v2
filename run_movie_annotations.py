
import pandas as pd
import numpy as np
import os
import sys
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import socket

import _mov_anno_1_emotion 

base_path =  "/Users/sweis/Data/Arbeit/Juseless/data/project/brainvar_sexdiff_movies/" 
results = "results_annotations"
movies = ["DD", "DMW", "DPS", "FG", "LIB", "S", "SS", "TGTBTU"]

TR = 0.980  # seconds

_mov_anno_1_emotion.main(base_path,movies,results,TR)
