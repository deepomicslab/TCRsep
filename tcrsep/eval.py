import numpy as np
import logging
logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger(__name__)
logging.getLogger("tape").setLevel(logging.WARNING)
from tcrsep.estimator import TCRsep
from tcrsep.dataset import *
from tcrsep.utils import *
import os
import argparse
import gzip
from tcrsep.pgen import Generation_model

def load_model(gen_path,sel_path,alpha=0.1,dropout=0.1,simulation=False):
    if gen_path =='None':
        package_path = inspect.getfile(tcrsep)
        gen_path = package_path.split('__init__.py')[0] + 'models/generation_model/CMV_whole'
    gen_model = Generation_model(gen_path)

    if sel_path =='None':
        package_path = inspect.getfile(tcrsep)
        sel_path = package_path.split('__init__.py')[0] + 'models/selection_model/CMV_whole.pth'

    sel_model = TCRsep(dropout=dropout,alpha=alpha,load_path = sel_path,gen_model_path=gen_path,simulation=simulation)
    return gen_model,sel_model

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')    
    parser.add_argument('--data_path',type=str)
    parser.add_argument('--data_emb_path',type=str,default='None')
    parser.add_argument('--sel_model_path',type=str,default='None')
    parser.add_argument('--save_dir',type=str,default='result_eval/')
    parser.add_argument('--gen_model_path',type=str,default='None')
    parser.add_argument('--alpha',type=float,default=0.1)        
    parser.add_argument('--simulation',default=False,action='store_true')    
    args = parser.parse_args()

    if not os.path.exists(args.save_dir):
        os.mkdir(args.save_dir) 
    save_seq_path = os.path.join(args.save_dir,'query_data.csv')
    save_emb_path = os.path.join(args.save_dir,'query_data_embedding.npy.gz')

    sep = ',' if '.csv' in args.data_path else '\t'
    df = pd.read_csv(args.data_path,sep=sep)
    samples = df[['CDR3.beta','V','J']].values
    if 'full_seq' not in df.columns:            
        full_seqs = cdr2full(samples,multi_process=True)  #v-j-cdr3
        if type(full_seqs[0]) != str:
            full_seqs = [c.decode('utf-8') for c in full_seqs]
        seqs = [[s[0] for s in samples],full_seqs]            
        df['full_seq'] = full_seqs
    else :
        seqs = [[cdr3 for cdr3 in df['CDR3.beta'].values],df['full_seq'].values]
    
    if args.data_emb_path == 'None':
        emb_model_path = None #use the default embedding model!
        emb = get_embedded_data(seqs,emb_model_path)                
    else:
        f = gzip.GzipFile(args.data_emb_path, "r")
        emb = np.load(f)
        logger.info(f'Done loading embeddings from {args.data_emb_path}')

    f = gzip.GzipFile(save_emb_path, "w") 
    np.save(file=f, arr=emb)
        
    gen_model,sel_model = load_model(args.gen_model_path,args.sel_model_path,alpha=args.alpha,simulation=args.simulation)        
    sel_factors= sel_model.predict_weights(emb)

    df['sel_factors'] = sel_factors
    if 'pgen' not in df.columns:
        logger.info('Begin computing pgens using OLGA')
        pgen = gen_model.p_gen(samples)
        logger.info('Done!')
        df['pgen'] = pgen
    df['ppost'] = df['pgen'].values * df['sel_factors'].values
    df.to_csv(save_seq_path,sep=sep,index=False)            