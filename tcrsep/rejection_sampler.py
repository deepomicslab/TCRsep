from scipy.stats import norm
import numpy as np
from copy import copy
from scipy.stats import pearsonr as pr 
from tcrsep.pgen import Generation_model
import pandas as pd 
import numpy as np
from tcrsep.estimator import *
from tcrsep.utils import cdr2full

def sampler(gen_model,estimator,N,c=10,multiple=10,tcrsep=True,emb_model_path=None):
    #base_dis: sample; pdf
    #estimator: predict_weights for generated samples from base_dis
    #transform: transform the seq data to emb that weight model can predict
    new_samples = []
    weights_new = []
    embeds = []
    while len(new_samples) < N:        
        num_left = N - len(new_samples)        
        num_gen = multiple * num_left
        samples = gen_model.sample(num_gen) # N x d    
        samples_ori = copy(samples) #only provide indexes
        if tcrsep:
            samples = get_embedded_data(samples,emb_model_path)
            # samples = transform(samples,model,model_cdr3,directory=gene_path) #transform to embedding
        u = np.random.uniform(size=len(samples)) #N us        
        weights = np.array(estimator.predict_weights(samples))
        weights[weights>1000] = 1000            
        accept = samples_ori[u <= (weights / float(c))]
        new_samples.extend(accept[:num_left])
        accept_weights = weights[u <= (weights / float(c))]
        weights_new.extend(accept_weights[:num_left])
        if tcrsep:
            accept_emb = samples[u <= (weights / float(c))]
            embeds.append(accept_emb)
    if tcrsep:
        embeds = np.concatenate(embeds,0)
        return np.array(new_samples),np.array(weights_new),embeds
    else :
        return np.array(new_samples),np.array(weights_new)