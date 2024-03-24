import numpy as np
from copy import copy 
import pandas as pd 
import numpy as np
from tcrsep.estimator import *
import argparse
from tcrsep.pgen import Generation_model
import logging
logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
import time

def sampler(gen_model,estimator,N,c=10,multiple=10,tcrsep=True,emb_model_path=None):
    #gen_model: generation model
    #estimator: predict_weights for generated samples from base_dis    
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
        ratio = len(new_samples) / N * 100
        ratio = round(ratio,3)
        logger.info(f"Done {ratio}%")
    if tcrsep:
        embeds = np.concatenate(embeds,0)
        return np.array(new_samples),np.array(weights_new),embeds
    else :
        return np.array(new_samples),np.array(weights_new)
    
def sampler_simulation(gen_model,estimator,N,c=10,multiple=10):    
    new_samples = []
    weights_new = []    
    while len(new_samples) < N:        
        num_left = N - len(new_samples)        
        num_gen = multiple * num_left
        samples = gen_model.sample(num_gen) # N x d    
        samples_ori = copy(samples) #only provide indexes
        u = np.random.uniform(size=len(samples)) #N us        
        weights = np.array(estimator.predict_weights(samples))
        weights[weights>1000] = 1000            
        accept = samples_ori[u <= (weights / float(c))]
        new_samples.extend(accept[:num_left])
        accept_weights = weights[u <= (weights / float(c))]
        weights_new.extend(accept_weights[:num_left])    
        ratio = len(new_samples) / N * 100
        ratio = round(ratio,3)
        logger.info(f"Done {ratio}%")    
    return np.array(new_samples),np.array(weights_new)
    
if __name__ == '__main__':    
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--simu_model',type=str,default='V_select')
    parser.add_argument('--gen_model_path',type=str,default='None')
    parser.add_argument('--pre_data_path',type=str,default='None')
    parser.add_argument('--save_path',type=str,default='data/simu.csv')
    parser.add_argument('--sample_num',type=int,default=1000000)
    parser.add_argument('--sel_factor_rel',type=float,default=2)
    parser.add_argument('--temp',type=float,default=0.1)        
    parser.add_argument('--seed',type=int,default=43)    
    parser.add_argument('--start_pos',type=int,default=3)
    parser.add_argument('--k',type=int,default=3)
    parser.add_argument('--sel_pos',type=int,default=0)
    parser.add_argument('--fre_thre',type=float,default=0.005)    
    args = parser.parse_args()

    dir = args.gen_model_path    
    if args.gen_model_path == 'None':
        package_path = inspect.getfile(tcrsep)
        gen_path = package_path.split('__init__.py')[0] + 'models/generation_model/CMV_whole'
        logger.info("Using default generation model.")
    else :
        gen_path = args.gen_model_path
    gen_model = Generation_model(gen_path)

    if args.pre_data_path == 'None':
        logger.info('Using generation model to generate pre-sel TCRs.')
        samples_pre = gen_model.sample(num=args.sample_num)
    else :
        samples_pre = np.array(pd.read_csv(args.pre_data_path)[['CDR3.beta','V','J']])

    if args.simu_model == 'V_select':
        sel_model = V_select(samples_pre,seed=args.seed,temp=args.temp)
    elif args.simu_model == 'J_select':
        sel_model = J_select(samples_pre,seed=args.seed,temp=args.temp)
    elif args.simu_model == 'VJ_select':
        sel_model = VJ_select(samples_pre,seed=args.seed,temp = args.temp)
    elif args.simu_model == 'Motif_select':            
        sel_model = Motif_select(samples_pre,start_pos=args.start_pos,k=args.k,
                                sel_factor=args.sel_factor_rel,sel_pos=args.sel_pos,thre=args.fre_thre)
    time0 = time.time()
    logger.info("Begin generating simulated post-selection TCRs")
    samples_post,ws = sampler_simulation(gen_model,sel_model,args.sample_num)
    time_used = time.time() - time0
    logger.info(f'Done generation! Used {time_used} seconds.')          
    res = pd.DataFrame({'CDR3.beta':[s[0] for s in samples_post],'V':[s[1] for s in samples_post],'J':[s[2] for s in samples_post],'sel':ws})                
    if args.save_path != 'None':
        if args.save_path.endswith('.csv'):
            res.to_csv(args.save_path,index=False)            