import numpy as np
from tcrsep.estimator import TCRsep
from tcrsep.dataset import *
import logging
from tcrsep.utils import *
import os
import argparse
import gzip
import json
logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')    
    parser.add_argument('--emb_model_path',type=str,default='None')
    parser.add_argument('--iters',type=int,default=10000)
    parser.add_argument('--batchsize',type=int,default=1000)
    parser.add_argument('--alpha',type=float,default=0.1)
    parser.add_argument('--val_ratio',type=float,default=0.1)    
    parser.add_argument('--post_data_path',type=str,default='None')
    parser.add_argument('--pre_data_path',type=str,default='None')
    parser.add_argument('--pre_emb_path',type=str,default='None')    
    parser.add_argument('--gen_model_path',type=str,default='None')
    parser.add_argument('--save_dir',type=str,default='None')    
    parser.add_argument('--simulation',default=False,action='store_true')
    args = parser.parse_args()

    if args.save_dir != 'None':
        if not os.path.exists(args.save_dir):
            os.mkdir(args.save_dir) 
        save_model_path = os.path.join(args.save_dir,'tcrsep.pth')
        save_pre_emb_path = os.path.join(args.save_dir,'pre_emb.npy.gz')
        save_post_emb_path = os.path.join(args.save_dir,'post_emb.npy.gz')
        save_args_path = os.path.join(args.save_dir,'args.json')
        save_pre_path = os.path.join(args.save_dir,'pre-sel.csv')
    else :
        save_model_path = None
        save_pre_emb_path=None
        save_post_emb_path=None
        save_args_path = None
        save_pre_path=None
    
    if args.pre_emb_path != "None":
        f = gzip.GzipFile(args.pre_emb_path, "r")
        gen_seqs = np.load(f)
        logger.info('Done loading embeddings for pre-selection clonetypes')
    else :
        gen_seqs = None

    if gen_seqs is None:
        if args.pre_data_path != "None":
            sep = ',' if '.csv' in args.pre_data_path else '\t'
            pre_df = pd.read_csv(args.pre_data_path,sep=sep)
            gen_seqs = np.array(pre_df[['CDR3.beta','V','J']])
    
    if args.post_data_path == "None":
        assert False, 'Please provide post-selection data!'
    if args.post_data_path.endswith('.npy.gz'):
        f = gzip.GzipFile(args.post_emb_path, "r")
        post_seqs = np.load(f)
        logger.info('Done loading embeddings for post-selection clonetypes')
    else:
        sep = ',' if '.csv' in args.post_data_path else '\t'
        post_seqs = pd.read_csv(args.post_data_path,sep=sep)
        post_seqs = post_seqs[['CDR3.beta','V','J']].values

    emb_model_path = None if args.emb_model_path == 'None' else args.emb_model_path
    sel_model = TCRsep(alpha=args.alpha ,gen_model_path=args.gen_model_path,simulation=args.simulation,emb_model_path=emb_model_path)
    
    seqs_pre,pre_emb,post_emb = sel_model.train(args.iters,post_seqs,gen_seqs,args.batchsize,save_model_path,args.val_ratio)

    if save_pre_emb_path is not None:
        f = gzip.GzipFile(save_pre_emb_path, "w") 
        np.save(file=f, arr=pre_emb)
    if save_post_emb_path is not None:
        f = gzip.GzipFile(save_post_emb_path, "w") 
        np.save(file=f, arr=post_emb)
    if save_args_path is not None:
        with open(save_args_path,'wt') as f:
            json.dump(vars(args),f,indent=4)
    if args.pre_emb_path == 'None' and save_pre_path is not None:
        pre_df = pd.DataFrame({'CDR3.beta':[s[0] for s in seqs_pre],'V':[s[1] for s in seqs_pre],'J':[s[2] for s in seqs_pre]})
        pre_df.to_csv(save_pre_path,index=False)
    
    


