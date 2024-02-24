from collections import defaultdict
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import os
import pickle
from scipy.stats import ttest_ind as ttest
import os
import sys
import inspect
import logging
from tqdm import tqdm
from tcrsep.sharing_analysis import DATCR,Sharing

#done
# sharing_predictor = Sharing('data/sharing')
# sharing_pre,sharing_real = sharing_predictor.predict_sharing('results/test/query_data.csv',get_actual_sharing=True)
# spectrum_pre,spectrum_real = sharing_predictor.sharing_spectrum(gen_model_path='models/generation_model/human_T_beta',sel_model_path='results/test2/tcrsep.pth' ,est_num=100000) 

# DATCR_predictor = DATCR('data/sharing')
# pvalues = DATCR_predictor.pvalue('results/test/query_data.csv')

from tcrsep.estimator import TCRsep
sel_model = TCRsep(default_sel_model=True)
query_tcrs = [['CASTQKPSYEQYF','TRBV6-9','TRBJ2-7'], ['CARGPYNEQFF','TRBV6-9','TRBJ2-1']]
sel_factors = sel_model.predict_weights(query_tcrs) #obtain selection factors
pgens, pposts = sel_model.get_prob(query_tcrs) #obtain pre- and post-selection probs 

# draw samples from p_post
post_samples = sel_model.sample(n=10)

from tcrsep.sharing_analysis import Sharing, DATCR
sharing_predictor = Sharing('data/sharing')

# predict sharing numbers of TCRs in query_data.csv among reps in the folder, "data/sharing"
sharing_pre,sharing_real = sharing_predictor.predict_sharing('data/query_data_evaled.csv') 

# predict the sharing spectrum for reps in "data/sharing"
spectrum_pre,spectrum_real = sharing_predictor.sharing_spectrum(est_num=10000) 

# identify DATCRs
DATCR_predictor = DATCR('data/sharing')
pvalues = DATCR_predictor.pvalue('data/query_data_evaled.csv')