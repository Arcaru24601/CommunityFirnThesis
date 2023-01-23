# -*- coding: utf-8 -*-
"""
Created on Wed Jan 18 14:38:38 2023

@author: jespe
"""

import numpy as np
import pandas as pd
import scipy.interpolate as interpolate
from Cubic_smooth import smooth_data
import h5py as hf
from matplotlib import pyplot as plt
from d18O_read import get_interval_data_noTimegrid, find_start_end_ind

