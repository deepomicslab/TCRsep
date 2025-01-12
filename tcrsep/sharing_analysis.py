from collections import defaultdict
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import os
import os
import inspect
import logging
from tqdm import tqdm
from tcrsep.utils import *
from tcrsep.pgen import Generation_model
from tcrsep.estimator import TCRsep,load_tcrsep
from scipy.optimize import least_squares as LS

logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class Sharing_analysis:
    def __init__(self,rep_dir,sep=',',compression=False):        
        self.rep_dir= rep_dir
        paths= os.listdir(self.rep_dir)
        if len(paths) == 0:
            assert False, "No repertoire file in the provided directory!"
        # sep = ',' if 'csv' in paths[0] else '\t' #loading csv file or tsv file                
        paths = [os.path.join(rep_dir,p) for p in paths]
        self.reps = self.process_rep(paths,sep,compression)        
        self.sharing_info,self.sharing_spec_info = self.get_sharing(self.reps)
    
    def process_rep(self,paths,sep=',',compression=False):
        reps = []
        for p in paths:
            if p.endswith('.gz') or compression:
                d = pd.read_csv(p,sep=sep,compression='gzip')
            else :
                #print(p,sep)
                d = pd.read_csv(p,sep=sep)            
            if 'amino_acid' in d.columns:
                d = d.rename(columns={'amino_acid':'CDR3.beta','v_gene':'V','j_gene':'J'})
            reps.append(d)
        reps = [np.array(reps[i][['CDR3.beta','V','J']]) for i in range(len(reps))]
        reps = [set([tuple(item) for item in rep]) for rep in reps]
        return reps

    def get_sharing(self,clone_types):
        #get the sharing spectrum; number -> number
        ref = defaultdict(int)
        clone_types = [set(cdr) for cdr in clone_types]
        for clone_set in clone_types:
            for c in clone_set:
                ref[c] += 1
        res = defaultdict(int)
        for c in ref.keys():
            res[ref[c]] += 1
        return ref,res
    
    def process_query(self,query_file,ppost_column='ppost'):
        query_pd = pd.read_csv(query_file)
        query_data = np.array(query_pd[['CDR3.beta','V','J']])
        query_data = [tuple(item) for item in query_data]
        if ppost_column in query_pd.columns:
            pposts = query_pd[ppost_column].values
        else :
            assert False, "Please eval the query file first;"
        return query_data,pposts
    
    def get_publicness(self,clonetypes):
        '''
        Get the sharing numbers of the input clonetypes
        '''
        return np.array([self.sharing_info[c] for c in clonetypes])

class DATCR(Sharing_analysis):
    def __init__(self,rep_dir,sep=',',compression=False):
        '''
        @rep_dir: the directory that containts repertoires
        '''
        super(DATCR,self).__init__(rep_dir,sep,compression)
        self.pposts = None
        # self.query_data = list(self.sharing_info.keys())
        # self.query_data = [c for c in self.query_data if self.sharing_info[c] >=self.sharing_thre] #whole candidate clonetypes

    def p_data(self,query_data,batch_num=100,sharings=None):
        # pdata = p_data_pos(self.reps,query_data)
        interval = len(query_data) // batch_num
        pdatas = []
        # for k in range(batch_num+1):
        if sharings is not None:
            already = dict()
            for k in range(len(query_data)):            
                query_data_sample = query_data[k]
                if sharings[k] not in already:
                    pdata = p_data_pos(self.reps,[query_data_sample])
                    pdata = [pdata[c] for c in [query_data_sample]]
                    already[sharings[k]] = pdata[0]
                    pdatas.append(pdata[0])
                else :
                    pdata = already[sharings[k]]
                    pdatas.append(pdata)
                if k % 5000 == 0:
                    r = round(len(pdatas) / len(sharings) * 100,3)
                    logger.info(f'Done {r}%')
        else :
            for k in range(len(query_data)):
                query_data_sample = query_data[k*interval:(k+1)*interval]
                if len(query_data_sample) == 0:
                    continue
                pdata = p_data_pos(self.reps,query_data_sample)
                pdata = [pdata[c] for c in query_data_sample]
                pdatas.extend(pdata)
                r = round((k+1) / batch_num * 100,3)
                logger.info(f'Done {r}%')
        return pdatas
    
    def pvalue(self,query_file,pdata=None,ppost_column='ppost',batch_num=100,rescale=True):
        '''
        Compute the pvalues of TCRs in the query_file regarding their publicness among the repertoires included in "rep_dir"
        @query_file: path to the query_file that contains the clonetypes and THE POST-SEL Probabilities (specified in the "ppost" column).
                     Please eval the query_file first if the "ppost" column is missing.
        Return:
            pvalues
        '''
        query_data,pposts = self.process_query(query_file,ppost_column)        
        num = len(query_data)
        if pdata is None:
            logger.info(f'Begin computing the P_data for {num} clonetypes')
            pdata = self.p_data(query_data)
            logger.info('Done computing p_data')
        logger.info('Begin computing pvalues')
        #print(pposts)                  
        lambda_ = LS(lambda x: np.sum((np.log(pdata) - np.log(pposts) - np.log(x))**2),x0=2.0)['x'][0]
        if rescale:
            logger.info(f'lambda : {lambda_}')
            pposts = pposts * lambda_

        if len(query_data) < 10000:
            prepare = []
            for i in range(len(query_data)):
                c = query_data[i]
                p = pposts[i]
                clone = c
                n1s,n2s = [],[]
                for rep in self.reps:
                    if clone in rep:
                        n1s.append(len(rep))
                    else :
                        n2s.append(len(rep))
                prepare.append((p,n1s,n2s))                                
            pvalues = p_val_pos(prepare)
        else :
            interval = len(query_data) // batch_num
            pvalues = []
            for k in range(batch_num+1):
                query_data_sample = query_data[k*interval:(k+1)*interval]
                ppost_sample = pposts[k*interval:(k+1)*interval]
                if len(query_data_sample) == 0:
                    continue
                prepare = []
                for i in range(len(query_data_sample)):
                    c = query_data_sample[i]
                    p = ppost_sample[i]
                    clone = c
                    n1s,n2s = [],[]
                    for rep in self.reps:
                        if clone in rep:
                            n1s.append(len(rep))
                        else :
                            n2s.append(len(rep))
                    prepare.append((p,n1s,n2s))  
                pvalues.extend(p_val_pos(prepare))
                r = round((k+1) / batch_num * 100,3)
                logger.info(f'Done {r}%')
        # pvalues = []
        # if len(prepare) > 10000:
        #     for i in range(batch_num+1):
        #         prepare_samples = prepare[i*interval:(i+1)*interval]
        #         if len(prepare_samples) == 0:
        #             continue
        #         pvalues.extend(p_val_pos(prepare_samples))                
        # else:
        #     pvalues = p_val_pos(prepare)
        logger.info('Done computing pvalues')

        return pvalues
    
class Sharing(Sharing_analysis):
    def __init__(self,rep_dir):
        super(Sharing,self).__init__(rep_dir)

    def predict_sharing(self,query_file,get_actual_sharing=True):
        '''
        Predict the sharing numbers of TCRs specified in the query_file
        @query_file: path to the query_file that contains the clonetypes and the corresponding post-sel probs.
        @get_actual_sharing: if set to True, will also return the actual sharing numbers of TCRs.
        Return:
            predicted_sharing_numbers, actual_sharing_numbers (when get_actual_sharing=True) 
            or  predicted_sharing_numbers (when get_actual_sharing=False)
        '''
        query_data,pposts = self.process_query(query_file)
        Nis = [len(rep) for rep in self.reps]
        sharings = []
        for i in tqdm(range(len(pposts))):
            c1 = 1 - (1 - pposts[i]) ** Nis          
            c1 = sum(c1)           
            sharings.append(c1)  
        if get_actual_sharing:
            sharing_real = [self.sharing_info[c] for c in query_data]
            return sharings, sharing_real
        else :
            return sharings
    
    def sharing_spectrum(self,sel_model_path=None,est_num=1000000):
        '''
        Predict the sharing spectrum. Will generate est_num of TCRs from P_post and compute the sharing spectrum using the generating function.
        @gen_model_path: path to the directory of the generation model. If not specified, will use the default generation model.
        @sel_model_path: path to the selection model. If not specified, will use the default selection model.
        @est_num: the number of TCRs to be generated to estimate the integral in the generating function.
        Return:
            A dictionary recording the predicted sharing spectrum, a dictionary recording the actual sharing spectrum
        '''
        tcrsep_model = load_tcrsep(sel_model_path)
        #rejection sampler
        logger.info('Begin generating synthetic post-selection samples.')
        syn_samples,weights = tcrsep_model.sample(N=est_num)
        logger.info('Done generating synthetic post-selection samples; begin computing their genration probs.')
        pgens,pposts = tcrsep_model.get_prob(syn_samples)
        logger.info('Done computing generation probs; Begin computing the sharing spectrum.')

        Nis = [len(rep) for rep in self.reps]
        est_sharing = share_num(pposts,Nis,600) #600 is a good choice
        sharing_dic = {i:est_sharing[i] for i in range(1,len(Nis)+1)}
        return sharing_dic,self.sharing_spec_info    

    # def load_model(self,gen_path,sel_path):
    #     if gen_path is None or gen_path == 'CMV':
    #         package_path = inspect.getfile(tcrsep)
    #         gen_path = package_path.split('__init__.py')[0] + 'models/generation_model/CMV_whole'
    #     elif gen_path == 'COVID19':
    #         package_path = inspect.getfile(tcrsep)
    #         gen_path = package_path.split('__init__.py')[0] + 'models/generation_model/COVID19'
    #     gen_model = Generation_model(gen_path)

    #     if sel_path is None or sel_path == 'CMV':
    #         package_path = inspect.getfile(tcrsep)
    #         sel_path = package_path.split('__init__.py')[0] + 'models/selection_model/CMV_whole.pth'
    #     elif sel_path == 'COVID19':
    #         package_path = inspect.getfile(tcrsep)
    #         sel_path = package_path.split('__init__.py')[0] + 'models/selection_model/COVID19.pth'

    #     sel_model = TCRsep(load_path = sel_path,gen_model_path=gen_path)
    #     return gen_model,sel_model



            