from tqdm import tqdm
import torch
import numpy as np
from Bio import SeqIO
from Bio import pairwise2
import os 
import multiprocessing as mp 
from math import ceil
from collections import defaultdict
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA as PCA
import scipy.integrate as integrate
import scipy.special as sc
from scipy.stats import betabinom as BB
from scipy.optimize import minimize
import math
import inspect
import tcrsep
import logging
logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger(__name__)
logging.getLogger("tape").setLevel(logging.WARNING)
from tcr2vec.model import TCR2vec
from tcrsep.dataset import *
import logging
from sympy import expand, symbols
import mpmath
from torch.utils.data import DataLoader

AAS = 'ACDEFGHIKLMNPQRSTVWY'

def com_ngram(cdr3s,n=3):
    res = defaultdict(int)
    for cdr3 in cdr3s:
        if len(cdr3) < n + 3:
            continue
        cdr3 = cdr3[2:-1]
        for i in range(len(cdr3)-n+1):
            res[cdr3[i:i+n]] += 1
    sum_ = sum([res[k] for k in res.keys()])
    for k in res.keys():
        res[k] /= sum_
    return res

def get_emb(tcrvec,loader,pool_method='mean',only_pool=False):
    '''
    Get the embeddings from TCRvec model
    @pool_method: mean or cls
    @only_pool: if set to True, no projection and will output the features of the BERT model.
    '''
    emb = []
    tcrvec.eval()
    with torch.no_grad():
        for batch in tqdm(loader):
            batch['input_ids'] = batch['input_ids'].to('cuda:0')
            batch['input_mask'] = batch['input_mask'].to('cuda:0')                     
            #self.model.l2_normalized=False
            emb_b = tcrvec(batch)                
            #emb_b = emb_b.detach().cpu().numpy() #b x emb
            emb.append(emb_b)                                
    emb = np.concatenate(emb)
    return emb

def rename_Vseg(Vname):
    if Vname[1] == 'C':
        Vname = 'TRB' + Vname[4:]
        if Vname[Vname.find('-') + 1] == '0':
            Vname = Vname[:(Vname.find('-') +
                            1)] + Vname[(Vname.find('-') + 2):]
        if Vname[Vname.find('V') + 1] == '0':
            Vname = Vname[:(Vname.find('V') +
                            1)] + Vname[(Vname.find('V') + 2):]
    return Vname

def rename_Jseg(Jname):
    if Jname[1] == 'C':
        Jname = 'TRB' + Jname[4:]

        if Jname[Jname.find('-') + 1] == '0':
            Jname = Jname[:(Jname.find('-') +
                            1)] + Jname[(Jname.find('-') + 2):]
        if Jname[Jname.find('J') + 1] == '0':
            Jname = Jname[:(Jname.find('J') +
                            1)] + Jname[(Jname.find('J') + 2):]
    return Jname

def map_gene(gene_name,ref_gene):
    #Hierachy mapping TRBVx/TRBJx
    ref_gene = ref_gene.split('|')[1]
    if gene_name == ref_gene :
        return True
    if str.isnumeric(gene_name[5:6]):
        gene_family = gene_name[4:6]
    else:
        gene_family = gene_name[4:5]
    if str.isnumeric(ref_gene[5:6]):
        ref_family = ref_gene[4:6]
    else:
        ref_family = ref_gene[4:5]
    if gene_family != ref_family:      
        return False

    if gene_name in ref_gene:
        return True
    if len(gene_name.split('-')) > 1:
       gene_split = gene_name.split('-')[1] 
       gene_f2 = gene_split[0]
       if len(ref_gene.split('-')) == 1:
           return False #ref has no allele
       if ref_gene.split('-')[1][0] != gene_f2:
           return False
    # if family level
    return True
    

def to_full_seq(directory, Vname, Jname, CDR3):
    ## Translate segment name into segment sequence
    foundV = False
    foundJ = False
    Vseq = ''
    Jseq = ''
    for Vrecord in SeqIO.parse(
        os.path.join(directory, 'V_segment_sequences.fasta'), "fasta"
    ):
        if type(Vname) != str or Vname == 'unresolved':
            print('Vname not string but ', Vname, type(Vname))
            Vseq = ''

        else:
            ## Deal with inconsistent naming conventions of segments
            Vname_adapted = rename_Vseg(Vname)     
            if not 'TRBV' in Vrecord.id:                
                continue
            # if Vname_adapted in Vrecord.id:            
            if map_gene(Vname_adapted,Vrecord.id):
                Vseq = Vrecord.seq
                foundV = True                        
            # elif '-' in Vname_adapted:                
            #     Vname_adapted = Vname_adapted.split('-')[0]                
            #     if Vname_adapted in Vrecord.id:
            #         Vseq = Vrecord.seq
            #         foundV = True                    
        if foundV:
            break   
    # print(foundV)             
    for Jrecord in SeqIO.parse(
        os.path.join(directory, 'J_segment_sequences.fasta'), "fasta"
    ):
        if type(Jname) != str or Jname == 'unresolved':
            print('Jname not string but ', Jname, type(Jname))
            Jseq = ''
        else:
            ## Deal with inconsistent naming conventions of segments
            Jname_adapted = rename_Jseg(Jname)
            if not 'TRBJ' in Jrecord.id:
                continue
            # if Jname_adapted in Jrecord.id:
            if map_gene(Jname_adapted,Jrecord.id):
                Jseq = Jrecord.seq
                foundJ = True
        if foundJ:
            break
    if foundV and Vseq != '':
        ## Align end of V segment to CDR3
        alignment = pairwise2.align.globalxx(
            Vseq[-5:],  # last five amino acids overlap with CDR3
            CDR3,
            one_alignment_only=True,
            penalize_end_gaps=(False, False)
        )[0]
        best = list(alignment[1])

        ## Deal with deletions
        if best[0] == '-' and best[1] == '-':
            best[0] = Vseq[-5]
            best[1] = Vseq[-4]
        if best[0] == '-':
            best[0] = Vseq[-5]

        # remove all left over -
        best = "".join(list(filter(lambda a: a != '-', best)))
    else:
        best = CDR3

    ## Align CDR3 sequence to start of J segment
    if Jseq != '':
        alignment = pairwise2.align.globalxx(
            best,
            Jseq,
            one_alignment_only=True,
            penalize_end_gaps=(False, False)
        )[0]

        # From last position, replace - with J segment amino acid
        # until first amino acid of CDR3 sequence is reached
        best = list(alignment[0])[::-1]
        firstletter = 0
        for i, aa in enumerate(best):
            if aa == '-' and firstletter == 0:
                best[i] = list(alignment[1])[::-1][i]
            else:
                firstletter = 1

        # remove all left over -
        best = "".join(list(filter(lambda a: a != '-', best[::-1])))

    full_sequence = Vseq[:-5] + best

    return full_sequence, foundV, foundJ

def divide_chunks(samples, n):     
    # looping till length l
    for i in range(0, len(samples), n):
        yield samples[i:i + n]

def cdr2full(samples,verbose=False,multi_process=False):
    package_path = inspect.getfile(tcrsep)
    directory = package_path.split('__init__.py')[0] + 'gene_segment/'
    if multi_process:
        if len(samples) > 5e6:
            logger.info('Using batched due to large data size')
            sample_ref = samples
            batch_num = len(samples) // 1000000 + 1 if len(samples) % 1000000 != 0 else len(samples) // 1000000
            full_seqs = []
            for i in range(batch_num):
                logger.info('Total batch: ',batch_num,'current batch: ',i+1)
                samples = sample_ref[i * 1000000:(i+1)*1000000]
                processes = mp.cpu_count() 
                #print(len(samples))
                n = ceil(len(samples) / processes)                  
                samples_chunks = list(divide_chunks(samples, n))
                args = [(directory,s) for s in samples_chunks]
                pool = mp.Pool(processes=processes)        
                full_seq = pool.map(_wrapper,args)
                full_seq = np.concatenate(full_seq)
                pool.close()
                pool.join()             
                full_seqs.append(full_seq)
            full_seqs = np.concatenate(full_seqs)
        else :     
            processes = mp.cpu_count() 
            #print(len(samples))
            n = ceil(len(samples) / processes)                  
            samples_chunks = list(divide_chunks(samples, n))
            args = [(directory,s) for s in samples_chunks]
            pool = mp.Pool(processes=processes)        
            full_seqs = pool.map(_wrapper,args)
            full_seqs = np.concatenate(full_seqs)
            pool.close()
            pool.join()                        
    else :
        full_seqs = _cdr2full(directory,samples,verbose)
    return full_seqs

def _wrapper(args):
   return _cdr2full(*args)

def _cdr2full(directory,samples,verbose=False):
    full_seq = []
    if not verbose:
        tqdm = list
    else :
        from tqdm import tqdm
    for sample in tqdm(samples):
        full,foundV,_ = to_full_seq(directory,str(sample[1]), str(sample[2]), str(sample[0]))        
        if foundV:
            full_seq.append(full._data)        
        else :
            full_seq.append('Failure')         
    return full_seq

def Entropy(p_pre,qs=None,base=2,eps=1e-20,sampled=True):
    if base == 2:
        log_fun = np.log2
    else:
        log_fun = np.log
    p_pre = np.array(p_pre)
    p_pre[p_pre==0] = eps
    if sampled:        
        if qs is None:
            return -np.mean(log_fun(p_pre))
        else :
            qs,p_pre = np.array(qs),np.array(p_pre)
            p_pre = log_fun(qs * p_pre) * qs
            return -np.mean(p_pre)
    else :
        return -np.sum(np.array(p_pre) * log_fun(p_pre))

def JS_div(input:dict,mode='pre_pre',base=2,eps=1e-30):
    #mode list: pre_pre, post_pre
    #           post_post model -> dataset
    if base == 2:
        log_fun = np.log2
    else:
        log_fun = np.log
    if mode == 'pre_pre':
        kl1 = 0.5 * np.mean(log_fun(input['p11']/(0.5 * input['p11'] + 0.5 * input['p21'])))
        kl2 = 0.5 * np.mean(log_fun(input['p22']/(0.5 * input['p12'] + 0.5 * input['p22'])))
        return kl1 + kl2
    elif mode == 'post_pre':
        array = input['qs'] * log_fun(input['qs'] / (0.5 * input['qs'] + 0.5)) - log_fun(0.5*input['qs'] + 0.5)
        return 0.5 * np.mean(array)
    elif mode == 'post_post_intra':
        kl1 = 0.5 * np.mean(log_fun(2*input['p11']/(input['p11']+input['p21'])))
        kl2 = 0.5 * np.mean(log_fun(2*input['p22'] / (input['p12']+input['p22'])))
    elif mode == 'post_post_intra_ori':
        kl1 = 0.5 * np.mean(input['qs11'] * log_fun(input['qs11'] * input['p11']/ 
                            (0.5 * input['qs11']* input['p11'] + 0.5 * input['qs21'] * input['p21'] + eps)))
        kl2 = 0.5 * np.mean(input['qs22'] * log_fun(input['qs22'] * input['p22']/ 
                            (0.5 * input['qs12']* input['p12'] + 0.5 * input['qs22'] * input['p22'] + eps)))
        return kl1 + kl2
    
    elif mode == 'post_post_inter_ori':
        half = 0.5 * (input['qs1'] + input['qs2'])
        kl = input['qs1'] * log_fun(input['qs1'] / half) + input['qs2'] * log_fun(input['qs2'] / half)
        return 0.5 * np.mean(kl)
    
    elif mode == 'post_post_inter':
        #samples are from p_post        
        kl1 = 0.5 * np.mean(log_fun(2*input['qs11'] / (input['qs11']+input['qs21'])))
        kl2 = 0.5 * np.mean(log_fun(2*input['qs22'] / (input['qs12']+input['qs22'])))        
        return kl1 + kl2
    else :
        raise Exception("no such mode")

def Gene_dis(gs,names=None):
    if names is None:
        names = list(set(gs))
        names.sort()
    names_ref = set(names)
    res = defaultdict(int)
    sum_ = 0
    for g in gs:
        if g not in names_ref:
            continue
        sum_ += 1
        res[g] += 1
    #normalize to 1?
    for g in names:
        res[g] /= len(gs)
    
    return [res[g] for g in names],names
def plot_gene_dis(data,g_names,labels,top_gene = 30,interval = 0.15,fig_name=None,fig_size=(12,4)):
    '''
    each input should be in the format of (mean_freqs,errs)
    also should be one-to-one with the gene list
    '''   
    plt.figure(figsize=fig_size)
    if len(data[0]) == 2:
        order=np.argsort(data[0][0])[::-1] #same gene (x_axis)
    else :
        order = np.argsort(data[0])[::-1]
    x_axis = np.array(list(range(top_gene)))
    num_to_plot = len(data)
    #plt.errorbar(x_axis-0.2, np.array(data[0])[order] , fmt='b+', linewidth=4, capsize=5,label='Data',markersize=8)
    for i in range(len(data)):
        axis = x_axis - (num_to_plot//2) * interval + i * interval
        if num_to_plot % 2 == 0:
            if - (num_to_plot//2) * interval + i * interval >= 0:
                axis+= interval
        if len(data[0]) == 2:
            plt.errorbar(axis, np.array(data[i][0])[order][:top_gene],np.array(data[i][1])[order][:top_gene] , fmt='o', linewidth=1, capsize=3,label=labels[i],markersize=3)        
        else :
            #print(np.array(data[i])[order])
            plt.errorbar(axis, np.array(data[i])[order][:top_gene], fmt='o', linewidth=1, capsize=3,label=labels[i],markersize=3)        
    plt.xticks(x_axis,np.array(g_names)[order][:top_gene])
    plt.xticks(rotation=45)
    plt.ylabel('Frequency',size=12)
    plt.xlabel('Gene',size=12)
    plt.grid()
    plt.legend()
    #plt.title('J USAGE DISTRIBUTION',fontsize=20)
    plt.tight_layout()
    if fig_name is not None:
        plt.savefig(fig_name,dpi=300)
    else: plt.show()

alphabet='ARNDCEQGHILKMFPSTWYV-'
# MAXLEN = 30
idx2aa = {i:alphabet[i] for i in range(len(alphabet))}

def tcrs2nums(tcrs):
    """Converts a list of (TCR) amino acid sequences to numbers. Each letter is changed to its index in the alphabet"""
    tcrs_num=[]
    n=len(tcrs)
    for i in range(n):
        t=tcrs[i]
        nums=[]
        for j in range(len(t)):
            nums.append(alphabet.index(t[j]))
        tcrs_num.append(nums)
    return tcrs_num

def add_gap(tcr,l_max,gap_char='-'):
    """Add gap to given TCR. Returned tcr will have length l_max.
    If there is an odd number of letters in the sequence, one more letter is placed in the beginning."""  
    l = len(tcr)
    if l<l_max:
        i_gap=np.int32(np.ceil(l/2))
        tcr = tcr[0:i_gap] + (gap_char*(l_max-l))+tcr[i_gap:l]
    return tcr

def check_align_cdr3s(cdr3s,lmaxtrain = 20):
    """Check cdr3s for too long sequences or sequences containing characters outside alphabet
    returns cdr3s_letter (proper cdr3s aligned, but improper sequences are left as they are)
            cdr3s_aligned (proper cdr3s aligned, places of improper sequences are left empty),
            and Ikeep3 (locations of proper cdr3s)
    Here improper means sequences that are longer than those in the training data or contain
    characters outside the used alphabet."""
    lmaxtest = lmaxtrain
    Ikeep3=np.ones((len(cdr3s),),dtype=bool)
    cdr3s_aligned=[]
    cdr3s_letter =[]

    for i in range(len(cdr3s)):
        ca = add_gap(cdr3s[i],lmaxtrain)
        cdr3s_aligned.append(ca)
        cdr3s_letter.append(ca)
    return cdr3s_letter, cdr3s_aligned, Ikeep3     

def l2_norm(emb):
    emb = emb / np.linalg.norm(emb,axis=1).reshape((len(emb),1))    
    return emb

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=5, verbose=True, delta=0, path='checkpoint.pt', trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print            
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func
    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            logger.info(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            logger.info(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss     

def likelihood(p,n1s,n2s):
    # print(p)
    if type(n1s) == list:
        n1s,n2s = np.array(n1s),np.array(n2s)
    a1 = 1 - np.exp(-p * n1s)
    # a1 = n1s * p * (1-p) ** (n1s-1)
    a2 = np.exp(-p * n2s)
    return np.prod(a1) * np.prod(a2)
def likelihood2(p,n1s,n2s):
    # print(p)
    if type(n1s) == list:
        n1s,n2s = np.array(n1s),np.array(n2s)
    a1 = 1 - np.exp(-p * n1s)
    # a1 = n1s * p * (1-p) ** (n1s-1)
    a2 = np.exp(-p * n2s)
    return np.prod(a1) * np.prod(a2)
def gradient(p,n1s,n2s):
    g = np.sum(n1s / (np.exp(p*n1s)-1)) - np.sum(n2s)
    return g

def ML_est(p0,n1s,n2s,lr=1e-15,thre=1e-15):
    #n1s,n2s = np.array(n1s),np.array(n2s)
    lh0 = likelihood(p0,n1s,n2s)    
    while True:
        p1 = p0 + gradient(p0,n1s,n2s) * lr
        # print(p1)
        # time.sleep(1)
        if np.abs(p1 - p0) <thre:
            break
        p0 = p1
  
    return p1    

def p_val(ppost,n1s,n2s):
    return integrate.quad(lambda x:likelihood(x,n1s,n2s),0,ppost)[0]

def log_comb(n,k):
    k_fac = k * np.log(k) - k if k > 100 else np.log(float(math.factorial(k)))
    n_fac = n * np.log(n) - n
    n_k_fac = (n-k) * np.log(n-k) - (n-k)
    return n_fac - k_fac - n_k_fac

from tqdm import tqdm
def p_data(reals,sharing_clones):
    res = dict()
    for clone in tqdm(sharing_clones.keys()):
        n1s,n2s = [],[]
        for real in reals:
            if clone in real:
                n1s.append(len(real))
            else :
                n2s.append(len(real))
        res[clone] = ML_est(1e-7,n1s,n2s)
    return res
#original 1e-4 end
def posterior(p,n1s,n2s):
    fenzi = likelihood(p,n1s,n2s)
    for i in range(3,7):
        if likelihood(10**(-i),n1s,n2s) > 0:
            break
    end = 10 ** (-(i-1))
    fenmu = integrate.quad(lambda x:likelihood(x,n1s,n2s),0,end)[0]
    return fenzi / fenmu

def MAE(start,end,nums,n1s,n2s):
    ps = 10 ** (-np.arange(start,end,(end-start)/nums))
    bestp = ps[0]
    bestvalue = posterior(bestp,n1s,n2s)
    for i in range(1,len(ps)):
        pos = posterior(ps[i],n1s,n2s)
        if pos > bestvalue:
            bestvalue = pos
            bestp = ps[i]
    return bestp


def p_data_pos(reals,clones,start=5,end=10,nums=250,multiprocess=True):
    res_pdata = dict()
    clone2ns = defaultdict(list)
    clones_new = []
    for clone in tqdm(clones):
        n1s,n2s = [],[]
        for real in reals:
            if clone in real:
                n1s.append(len(real))
            else :
                n2s.append(len(real))
        # clone2ns[clone].append(n1s)
        # clone2ns[clone].append(n2s)
        clones_new.append((clone,n1s,n2s))
    # clones_new = [(c,clone2ns[c][0],clone2ns[c][1]) for c in clones]
    #     res[clone] = MAE(start,end,nums,n1s,n2s)
    #print('done prepare')
    processes = mp.cpu_count()
    n = ceil(len(clones_new) / processes) 
    samples_chunks = list(divide_chunks(clones_new, n))
    pool = mp.Pool(processes=processes)
    res = pool.map(p_data_pos_,samples_chunks)
    pdata = np.concatenate(res)
    pool.close()
    pool.join()
    for i,c in enumerate(clones):
        res_pdata[c] = pdata[i]
    return res_pdata

def p_data_pos_(clones):
    res = []
    for clone in clones:
        n1s,n2s = clone[1],clone[2]
        res.append(MAE(5,10,200,n1s,n2s))
    return res

def p_val_pos(prepares):
    # clones_new = [(prepares[i],ns[i][0],ns[i][1]) for i in range(len(pposts))]
    clones_new = prepares
    processes = mp.cpu_count()
    n = ceil(len(clones_new) / processes) 
    samples_chunks = list(divide_chunks(clones_new, n))
    pool = mp.Pool(processes=processes)
    res = pool.map(p_val_pos_,samples_chunks)
    pvalues = np.concatenate(res)    
    return pvalues

def p_val_pos_(clones):
    res = []
    for clone in clones:
        res.append(integrate.quad(lambda x:posterior(x,clone[1],clone[2]),0,clone[0])[0])
    return res

def log_likelihood_rep(k,n,param):
    return BB.pmf(k,n,param[0],param[1])

def min_fun(param,ks,ns):
    theta = 1.0 / (param[0]+param[1])
    mu = param[0] * theta
    l = 0
    for i in range(len(ks)):        
        l+= np.log(np.arange(ks[i]) * theta + mu).sum() +np.log(np.arange(ns[i]-ks[i])*theta + 1 - mu).sum() - \
            np.log(np.arange(ns[i]) * theta + 1).sum()
    return -l

def grad_alpha(ks,ns,param):
    N = len(ns)
    a = -N * (sc.psi(param[0]) - sc.psi(param[0]+param[1]))
    b = np.sum(sc.psi([k+param[0] for k in ks]) - sc.psi([param[0]+param[1]+ns[i] for i in range(len(ks))]))
    return a + b

def grad_beta(ks,ns,param):
    N = len(ns)
    a = -N * (sc.psi(param[1])-sc.psi(param[0]+param[1]))
    b = np.sum(sc.psi([ns[i]-ks[i]+param[1] for i in range(len(ks))]) - 
               sc.psi([ns[i]+param[0]+param[1] for i in range(len(ks))]))
    return a + b

def find_param(ks,ns,param0,method='BFGS'):
    res = minimize(min_fun,param0,args=(ks,ns))
    return res

def com_aa_fre(cdr3s):
    res = defaultdict(list)
    for cdr3 in cdr3s:
        l = len(cdr3)
        if l % 2 == 0:
            inter = (l-2) // 2 #
            for i in range(-inter,inter):
                if i == -1 or i == 0:
                    res[0].append(cdr3[i + l//2])
                else :
                    if i < 0:
                        res[i + 1].append(cdr3[i + l//2]) 
                    else :
                        res[i].append(cdr3[i + l // 2])
        else :
            inter = (l-2) // 2
            for i in range(-inter,0):
                res[i].append(cdr3[l//2+ i])
            res[0].append(cdr3[l//2])
            for i in range(1,inter+1):
                res[i].append(cdr3[l//2+i])
    for k in res.keys():
        a = res[k]
        fres = []
        for AA in AAS:
            fres.append(len([aa for aa in a if aa == AA]) / len(a))
        res[k] = fres
    return res

def get_embedded_data(seqs,emb_model_path=None,L2_norm=True):

    if emb_model_path is None:
        package_path = inspect.getfile(tcrsep)
        emb_model_path = package_path.split('__init__.py')[0] + 'models/embedding_model/'
    #print(emb_model_path + '/' + 'TCR2vec/')
    emb_model = TCR2vec(emb_model_path + '/' + 'TCR2vec/').to('cuda:0')
    emb_model.eval()
 
    emb_model_cdr3 = TCR2vec(emb_model_path + '/' + 'CDR3vec/').to('cuda:0')
    emb_model_cdr3.eval()        

    if len(seqs) == 2 and len(seqs[1][0]) >=60 : #containing full seqs
        full_seqs = seqs[1]
        cdr3s = seqs[0]
    else :
        full_seqs = cdr2full(seqs,multi_process=True)
        if type(full_seqs[0]) != str:
                full_seqs = [c.decode('utf-8') for c in full_seqs]
        cdr3s = [s[0] for s in seqs]
        
    dset = TCRLabeledDset(full_seqs,only_tcr=True)
    loader = DataLoader(dset,batch_size=128,collate_fn=dset.collate_fn,shuffle=False) 
    logger.info('Encoding TCR-beta sequences.')
    emb = get_emb(emb_model,loader)  
    dset = TCRLabeledDset(cdr3s,only_tcr=True)
    loader = DataLoader(dset,batch_size=128,collate_fn=dset.collate_fn,shuffle=False) 
    logger.info('Encoding CDR3 sequences.')
    emb_cdr3 = get_emb(emb_model_cdr3,loader)
    del emb_model,emb_model_cdr3
    if L2_norm: #perform L2 nomalization
        emb = l2_norm(emb)
        emb_cdr3 = l2_norm(emb_cdr3)
    return np.concatenate((emb,emb_cdr3),1)

def func(x,q):
    return(x-np.log10(q))

def Likelihood(bin_min, bin_max, x, n):
    prod_x=[]
    bins = np.logspace(bin_min-1, bin_max, base=10, num=20)
    for p in bins:
        prod0 = 1
        for j in range(1,len(n)+1):
            if j in x:
                prod0 = p * n[j-1]*prod0 if p<1e-21 else (1 - float(mpmath.exp(-p * n[j-1])))*prod0 #approximation of small exp 1-e^-x = x
            else:
                prod0 = (1 - (p * n[j-1]))*prod0 if p<1e-21 else float((mpmath.exp(-p* n[j-1])))*prod0
        prod_x.append(prod0)
    return(prod_x, bins)

def share_num(ps,Nis,bins=50):
    #integral in log space
    ps = np.array(ps) + 1e-30
    # ps = np.array(ps)    
    # ps[ps == 0] = 1
    log_p = -np.log(ps)
    densities,bin_axis = np.histogram(log_p,bins=bins,density=True)
    # print(bin_axis)    
    # for i in range(1,len(bin_axis)):
    #     print(bin_axis[i] - bin_axis[i-1])
    x = symbols('x')  
    res = [0] * (len(Nis)+1)       
    for i in range(1,len(densities)+1):        
        if i % 20 == 0 and i != 0:
            curr = round((i+1) / bins * 100,2)
            logger.info(f'Complete {curr}%')
        #print(i)
        # interval = 10 ** bin_axis[i+1] - 10**bin_axis[i]
        # p = float(10 ** (-bin_axis[i]))
        p = np.exp(-bin_axis[i-1])
        # print(densities[i],bin_axis[i+1] - bin_axis[i],p,1/p)
        left = densities[i-1] * (bin_axis[i] - bin_axis[i-1]) / p
        right_exp = 1   
        # print(left)     
        # print(densities[i],p)
        for k in range(len(Nis)):            
            # print(p)            
            # if p < 1e-21:            
            #     right_exp = right_exp * 
            # if p < 1e-21:
            #     x1 = 1 - p * Nis[k]
            #     x2 = p * Nis[k]
            # else :
            #     x1 = float(mpmath.exp(-Nis[k] * p))
            #     x2 = 1 - float(mpmath.exp(-Nis[k] * p))
            inter = float(mpmath.exp(-Nis[k]*p))
            right_exp = right_exp * (inter + (1-inter)*x)
            # right_exp = right_exp * (x1 + x2* x)

        exp = expand(right_exp * left)
        # print('Eva time: ',time.time()-time0)
        for k in range(len(res)):
            res[k] += exp.coeff(x ** k)
    return res

def get_sharing(cdr3s):
    ref = defaultdict(int)
    cdr3s = [set(cdr) for cdr in cdr3s]
    for cdr3_set in cdr3s:
        for c in cdr3_set:
            ref[c] += 1
    res = defaultdict(int)
    for c in ref.keys():
        res[ref[c]] += 1
    return res

