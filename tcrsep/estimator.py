import torch
import torch.nn as nn   
import numpy as np
import logging
from tcrsep.dataset import Loader
from tcrsep.utils import *
from tcrsep.dataset import Loader
from tcrsep.pgen import Generation_model
from collections import defaultdict
from tqdm import tqdm
import json
from copy import copy
from torch.optim.lr_scheduler import ReduceLROnPlateau

logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def weights_init(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform(m.weight.data)

def load_tcrsep(load_dir=None,device='cuda:0'):
    default_model = False
    package_path = inspect.getfile(tcrsep)
    if load_dir is None or load_dir == 'None':
        default_model=True        
        load_dir = package_path.split('__init__.py')[0] + 'models/selection_model/CMV/'

    default_path = package_path.split('__init__.py')[0] + 'models/'
    with open(os.path.join(load_dir,'args.json'), 'r') as fp:
        args = json.load(fp)
        args = defaultdict(lambda: None,args)
    if not os.path.exists(args['gen_model_path']):
        #check the generation model
        if args['gen_model_path'] == 'models/generation_model/COVID/':        
            args['gen_model_path'] = default_path + 'generation_model/COVID/'
        elif args['gen_model_path'] == 'models/generation_model/CMV_whole/':
            args['gen_model_path'] = default_path + 'generation_model/CMV_whole/'

    sel_model = TCRsep(alpha=args['alpha'],gen_model_path=args['gen_model_path'],device=device,optimizer=args['optimizer'],lr=args['lr'])
    load_path = os.path.join(load_dir,f'tcrsep.pth')
    sel_model.model.load_state_dict(torch.load(load_path))
    sel_model.model = sel_model.model.to(device)
    sel_model.model.eval()
    if 'Z' in args:
        sel_model.Z = float(args['Z'])
    if default_model:
        logger.info(f'Loaded the default TCRsep model.')    
    else :
        logger.info(f'Loaded the TCRsep model from {load_path}.')
    return sel_model

class TCRsep:
    def __init__(self,alpha=0.1,device='cuda:0',gen_model_path=None,default_sel_model=False,optimizer='adam',lr=1e-4):         
        '''
        @alpha: the parameter alpha of TCRsep
        @device: the device name. Recommended specifying a GPU for acceralation.
        @load_path: If specified, will load the TCRsep model from the given path; Otherwise, will initialize a new model        
        @gen_model_path: path to the directory of the generation model; if not provided, will load the default generation model inferred on the Emerson data.
        @default_sel_model: if True, will load the default TCRsep model inferred on the Emerson data. Default is False.
        '''
        
        self.min_clip = -10
        self.max_clip = 5 #clip the output of Neural network; the same as soNNia and SONIA
        self.optimizer= optimizer
        self.model = NN().to(device)
        self.device= device   
        if default_sel_model:
            self.load_default_model()

        self.lr = lr
        self.alpha = alpha            
        self.default_gen_model = Generation_model(gen_model_path)    
        self.gen_model_path = gen_model_path
        self.emb_model_path = None

        self.Z = 1
    
    def fit(self,iters,loader,save_checkpoint=None,valid_emb=None,patience=10):
        self.model.train() 
        if self.optimizer== 'adam' :
            logger.info('Use Adam optimizer.')
            optimizer = torch.optim.Adam(self.model.parameters(),lr=self.lr)
        else :
            logger.info('Use RMSProp')
            optimizer = torch.optim.RMSprop(self.model.parameters(),lr=self.lr)

        scheduler = ReduceLROnPlateau(optimizer, 'min',patience=5, factor=0.2)       
        monitor=False
        if valid_emb is not None:
            valid_emb[0] = torch.FloatTensor(valid_emb[0]).to(self.device)
            valid_emb[1] = torch.FloatTensor(valid_emb[1]).to(self.device)            
            early_stop = EarlyStopping(patience=patience,path = save_checkpoint)
            monitor=True
        indicator = False
        e = 0
        while True:
            if indicator:
                break
            loss_train_record = []
            for i,batch in enumerate(loader):
                self.model.train()
                batch = [torch.FloatTensor(b).to(self.device) for b in batch]                          
                if i >= iters: #end training
                    break      
                if i == 200:  #check that the initialization is ok              
                    weights_pre1 = self.model(batch[0])
                    weights_pre1 = torch.clamp(weights_pre1,min=self.min_clip,max=self.max_clip)
                    weights_pre1 = torch.exp(weights_pre1)
                    if weights_pre1.mean().item() < 0.4:                  
                        self.model.apply(weights_init)
                        logger.info('Re-initialize the neural network due to a bad intialization.')
                        break
                    else:
                        indicator = True
                
                weights_post1 = self.model(batch[1]) #post-selection Qs     
                weights_post1 = torch.clamp(weights_post1,min=self.min_clip,max=self.max_clip)
                weights_post1 =  torch.exp(weights_post1)          
                weights_post1_alpha = weights_post1 / (self.alpha * weights_post1 + 1-self.alpha) #transform to Q_{alpha}
                weights_post1_alpha_post = 1 / ((1-self.alpha) * weights_post1 + self.alpha)

                weights_pre1 = self.model(batch[0]) #pre-selection Qs                           
                weights_pre1 = torch.clamp(weights_pre1,min=self.min_clip,max=self.max_clip)
                weights_pre1 = torch.exp(weights_pre1)
                weights_pre1_alpha = weights_pre1 / (self.alpha * weights_pre1 + 1-self.alpha)  #transform                               
                weights_pre1_alpha_post = 1 / ((1-self.alpha) * weights_pre1 + self.alpha)

                loss1 = 0.5 * self.alpha * ((weights_post1_alpha)**2).mean() + (1-self.alpha) *0.5 * ((weights_pre1_alpha)**2).mean() - (weights_post1_alpha).mean()
                loss2 = (weights_pre1.mean()-1.0)**2
                loss3 = 0.5 * self.alpha * ((weights_pre1_alpha_post)**2).mean() + (1-self.alpha) * 0.5 * ((weights_post1_alpha_post)**2).mean() - (weights_pre1_alpha_post).mean()                
                loss = loss1 + loss2 + loss3

                optimizer.zero_grad()
                loss.backward() 
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.1)               
                optimizer.step()
                loss_train_record.append(loss.item())
                
                if i // self.interval != e and indicator:
                    logger.info(f"At epoch {e}")
                    e = i // self.interval 
                    self.model.eval()
                    if valid_emb is not None:
                        #ws = self.predict_weights(valid_emb[0].detach().cpu().numpy())                                      
                        #ws_pre = self.predict_weights(valid_emb[1].detach().cpu().numpy()) #pre emb                         
                        ws = self.model(valid_emb[0])
                        ws_pre = self.model(valid_emb[1])
                        ws_pre = torch.clamp(ws_pre,min=self.min_clip,max=self.max_clip)
                        ws = torch.clamp(ws,min=self.min_clip,max=self.max_clip)
                        ws_pre = torch.exp(ws_pre).detach().cpu().numpy()
                        ws = torch.exp(ws).detach().cpu().numpy()                        
        

                        ws_alpha = ws / (self.alpha * ws + 1-self.alpha)            
                        ws_pre_alpha = ws_pre / (self.alpha * ws_pre + 1-self.alpha)                        
                        ws_pre_alpha_post = 1 / ((1-self.alpha) * ws_pre + self.alpha)
                        ws_alpha_post =  1 / ((1-self.alpha) * ws  + self.alpha)                        
                        val_loss = 0.5 * self.alpha * ((ws_alpha)**2).mean() + (1-self.alpha) *0.5 * ((ws_pre_alpha)**2).mean() - (ws_alpha).mean()
                        val_loss2 = 0.5 * self.alpha * ((ws_pre_alpha_post)**2).mean() + (1-self.alpha) * 0.5 * ((ws_alpha_post)**2).mean() - ws_pre_alpha_post.mean()             
                        val_loss += val_loss2
                        scheduler.step(val_loss) 
                        logger.info(f'At epoch {e}. ' +'Training loss = '+str(np.mean(loss_train_record)) + ', ' +  'Validation loss = '+ str(val_loss.item()) + '.')
                    else :
                        logger.info(f'At epoch {e}. ' +'Training loss = '+str(np.mean(loss_train_record)))
                    if monitor:
                        early_stop(val_loss, self.model)
                        if early_stop.early_stop:
                            logger.info("Early stopping")
                            if save_checkpoint is not None:
                                self.model.load_state_dict(torch.load(save_checkpoint))
                                logger.info('Saved and Loaded the best model')
                            break                   
                    if e % 10 == 0 and e > 0 and not monitor and save_checkpoint is not None:
                        sp = save_checkpoint.replace('tcrsep.pth',f'tcrsep_{e}.pth')
                        torch.save(self.model.state_dict(), sp)
                        

        if save_checkpoint is not None and valid_emb is None:
            torch.save(self.model.state_dict(), save_checkpoint)                    
            logger.info(f'At epoch {e} save the model to {save_checkpoint}.')

    def train(self,epochs,seqs_post,seqs_pre=None,batch_size=512,save_checkpoint=None,valid_ratio=0.1):
        '''
        @iters: iteration for the training process.
        @seqs_post: post-sel TCRs or their embedding numpy array. Its format should be [[CDR3_1,V_1,J_1],[CDR3_2,V_2,J_2],...], or in the shape of N x embedding_size.
        @seqs_pre: pre-sel TCRs or their embedding numpy array. It not provided, will generate pre-sel TCRs using the default generation model.
        @batch_size: the batch size for training.
        @save_checkpoint: the path to save the selection model.
        @valid_ratio: ratio of the training data that will be separated as validation data.
        Return:
            pre-sel sequences (or their embedding array), embedding array of pre-sel TCRs, embedding array of post-sel TCRs
        '''
        max_length = len(seqs_post) if seqs_pre is None else max(len(seqs_post),len(seqs_pre))
        iters = epochs * (max_length // batch_size)
        self.interval = max_length // batch_size
        
        if seqs_pre is None:
            n = len(seqs_post)            
            logger.info(f'Begin generating {n} pre-selection sequences using generation model from {self.default_gen_model.model_folder}.')
            seqs_pre = self.default_gen_model.sample(n)
            logger.info('Done!')
        
        logger.info('Transforming TCRs to embeddings.')
        if type(seqs_pre[0][0]) in [np.str_,str]: #need to get full TCR-beta
            seqs_pre_full = cdr2full(seqs_pre,multi_process=True)  #v-j-cdr3
            if type(seqs_pre_full[0]) not in [np.str_,str]:
                seqs_pre_full = [c.decode('utf-8') for c in seqs_pre_full]
            seqs_for_emb = [[s[0] for s in seqs_pre],seqs_pre_full]            
            pre_emb = get_embedded_data(seqs_for_emb,self.emb_model_path)
        else :
            pre_emb = seqs_pre 

        if type(seqs_post[0][0]) == str:
            seqs_post_full = cdr2full(seqs_post,multi_process=True)  #v-j-cdr3
            if type(seqs_post_full[0]) not in [np.str_,str]:
                seqs_post_full = [c.decode('utf-8') for c in seqs_post_full]
            seqs_post = [[s[0] for s in seqs_post],seqs_post_full]
            post_emb = get_embedded_data(seqs_post,self.emb_model_path)
        else:
            post_emb = seqs_post
        logger.info('Done embedding.')
        
        post_emb_ori = copy(post_emb)
        pre_emb_ori = copy(pre_emb)
        emb_valid= None
        if valid_ratio > 0:
            index = np.random.permutation(len(post_emb))
            index_pre = np.random.permutation(len(pre_emb))            
            post_emb_val = post_emb[index[:int(len(index) * valid_ratio)]]
            
            post_emb = post_emb[index[int(len(index) * valid_ratio):]]  
            pre_emb_val = pre_emb[index_pre[:int(len(index_pre) * valid_ratio)]]        
            pre_emb = pre_emb[index_pre[int(len(index_pre) * valid_ratio):]]                        
            emb_valid = [post_emb_val,pre_emb_val]
        loader_train = Loader(pre_emb,post_emb,batch_size)
        self.fit(iters,loader_train,save_checkpoint=save_checkpoint,valid_emb = emb_valid,patience=10) #patience can be changed to samller values
        self.Z = np.mean(self.predict_weights(pre_emb_ori))
        # if emb_valid is not None:
        #     self.Z_val = np.mean(self.predict_weights(pre_emb_val))        
        return seqs_pre,pre_emb_ori,post_emb_ori
    
    def load_default_model(self):        
        return load_tcrsep(device=self.device)

    def predict_weights(self,samples,batch_size=128,return_emb=False,emb=False):
        '''
        Predict the selection factors for input samples.
        @samples: input TCRs or their embedding numpy array.
        @batch_size: predict the selection factors in batches with size of @batch_size.
        @return_emb: will return both the predicted selection factors and the embedding of @samples
        Return:
            selection factors
        '''
        self.model.eval() 
        batch_num = len(samples) // batch_size +1
        weights_pre = []
        if type(samples[0][0]) in [str ,np.str_]:            
            samples = get_embedded_data(samples,self.emb_model_path)
        for i in range(batch_num):
            if len(samples[i*batch_size:(i+1)*batch_size]) == 0:
                continue
            samples_sub = samples[i * batch_size:(i+1)*batch_size]            
            weights_pre_tmp = self.model(torch.FloatTensor(samples_sub).to(self.device))  
            weights_pre.append(weights_pre_tmp.detach().cpu().numpy()[:,0])
        ws_pre = np.concatenate(weights_pre)
        ws_pre = np.clip(ws_pre,a_min=self.min_clip,a_max=self.max_clip)            
        ws_pre = np.exp(ws_pre)        
        ws_pre = ws_pre / self.Z                  
        if return_emb:
            return ws_pre,samples
        else :
            return  ws_pre
    
    def save(self,save_dir):
        #dir is the directory to save the model
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        save_model_path = os.path.join(save_dir,'tcrsep.pth')
        save_args_path = os.path.join(save_dir,'args.json')
        args = {'gen_model_path':self.gen_model_path,
                'alpha':str(self.alpha),
                'Z':str(self.Z),
                'save_dir':save_dir}
        if args['gen_model_path'] is None:
            args['gen_model_path'] = 'None'
        with open(save_args_path,'wt') as f:                       
            json.dump(args,f,indent=4) 

        torch.save(self.model.state_dict(), save_model_path)
         

    def get_prob(self,samples,pgens=None):
        '''
        Compute probabilities of the input samples
        @samples: in the format of [[CDR3_1,V_1,J_1],[CDR3_2,V_2,J_2],...]                   
        Return:
            pgen,ppost
        '''
        # if len(samples) == 2 and len(samples[0][0]) == 3: #[CDR3-V-J,embedding]
        sel_factors = self.predict_weights(samples)
        if pgens is None:            
            pgen = self.default_gen_model.p_gen(samples)        
        pposts = pgen * sel_factors
        return pgen,pposts
    
    def get_pgen(self,samples):
        pgen = self.default_gen_model.p_gen(samples)              
        return pgen

    def sample(self,N,c=10,multiple=5):
        '''
        @N: the number of TCRs sampled from P_post
        Return:
            Generated TCRs, selection factors of generated TCRs
        '''
        new_samples = []
        weights_new = []    
        logger.info('Begin sampling from P_post')
        while len(new_samples) < N:        
            num_left = N - len(new_samples)        
            num_gen = multiple * num_left
            samples = self.default_gen_model.sample(num_gen) # N x d    
            u = np.random.uniform(size=len(samples)) #N us        
            weights = np.array(self.predict_weights(samples))       
            accept = samples[u <= (weights / float(c))]
            new_samples.extend(accept[:num_left])
            accept_weights = weights[u <= (weights / float(c))]
            weights_new.extend(accept_weights[:num_left])            
            ratio = len(new_samples) / N * 100
            ratio = round(ratio,3)
            logger.info(f"Done {ratio}%")
        
        return np.array(new_samples),np.array(weights_new)

class NN(nn.Module):
    def __init__(self):
        super().__init__()
        in_dims = [128,128]
        self.in_dims = in_dims #embedding sizes of TCRbeta and CDR3beta
        hid_dim = 128
        out_dim=1
        act = nn.Tanh()         

        self.projects = nn.ModuleList([nn.Sequential(
            nn.Linear(in_dims[i], hid_dim),                 
            act,
            nn.Linear(hid_dim, hid_dim),
            act
        ) for i in range(len(in_dims))] )
        self.project3 = nn.Sequential(
            nn.Linear(hid_dim * len(in_dims), len(in_dims) * hid_dim // 2), 
            nn.Dropout(0.0),
            act,                
            nn.Linear(len(in_dims) * hid_dim // 2, 32), 
            nn.Dropout(0.0),
            act,                
            nn.Linear(32, out_dim))

    def forward(self, x):  
        xs = []
        idx = 0
        for i in range(len(self.in_dims)):
            xs.append(x[:,idx:idx+self.in_dims[i]])
            idx += self.in_dims[i]
        xs = [self.projects[i](xs[i]) for i in range(len(xs))]
        x = torch.cat(xs,-1)            
        x1  =self.project3(x)                
        return x1    