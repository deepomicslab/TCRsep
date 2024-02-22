import torch
import torch.nn as nn   
import numpy as np
import logging
from tcrsep.dataset import Loader
from tcrsep.utils import *
from tcrsep.dataset import Loader
from tcrsep.pgen import Generation_model
from tcrsep.rejection_sampler import sampler

logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def weights_init(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform(m.weight.data)

class TCRsep:
    def __init__(self,hidden_size=128,out_dim=1,dropout=0.0,device='cuda:0',
                 lr=1e-4,load_path=None,sizes=[128,128],alpha=0.1,simulation=False,gen_model_path=None,emb_model_path=None):         
        self.model = NN(sizes,hidden_size,out_dim,dropout,simulation=simulation).to(device)
        if load_path is not None:
            self.model.load_state_dict(torch.load(load_path))
            self.model.eval()
            logger.info(f'Load the trained model from {load_path}')
        self.sizes,self.hidden_size,self.out_dim,self.dropout = sizes,hidden_size,out_dim,dropout
        self.lr = lr        
        self.device= device               
        
        self.alpha = alpha            
        self.default_gen_model = Generation_model(gen_model_path)    
        self.emb_model_path = emb_model_path

    def check_and_reInitialize(self,ws):
        indicator = False
        if ws.mean().item() < 0.1:
            self.model.apply()
            indicator = True
        return indicator
    
    def fit(self,iters,loader,save_checkpoint=None,valid_emb=None,patience=5):
        self.model.train()           
        optimizer = torch.optim.Adam(self.model.parameters(),lr=self.lr)
        scheduler =torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=64)        
        if valid_emb is not None:
            valid_emb[0] = torch.FloatTensor(valid_emb[0]).to(self.device)
            valid_emb[1] = torch.FloatTensor(valid_emb[1]).to(self.device)            
            early_stop = EarlyStopping(patience=patience,path = save_checkpoint)
        indicator = False
        while True:
            if indicator:
                break
            for i,batch in enumerate(loader):
                self.model.train()
                batch = [torch.FloatTensor(b).to(self.device) for b in batch]                          
                if i >= iters: #end training
                    break      
                if i == 200:  #check that the initialization is ok              
                    weights_pre1,_ = self.model(batch[0])
                    if weights_pre1.mean().item() < 0.4:                  
                        self.model.apply(weights_init)
                        logger.info('Re-initialize the neural network')
                        break
                    else:
                        indicator = True
                weights_post1,_ = self.model(batch[1]) #post-selection Qs                  
                weights_post1_alpha = weights_post1 / (self.alpha * weights_post1 + 1-self.alpha) #transform to Q_{alpha}

                weights_pre1,_ = self.model(batch[0]) #pre-selection Qs                           
                weights_pre1_alpha = weights_pre1 / (self.alpha * weights_pre1 + 1-self.alpha)               

                loss1 = 0.5 * self.alpha * ((weights_post1_alpha)**2).mean() + (1-self.alpha) *0.5 * ((weights_pre1_alpha)**2).mean() - (weights_post1_alpha).mean()
                loss2 = (weights_pre1.mean()-1.0)**2
                loss = loss1 + loss2

                optimizer.zero_grad()
                loss.backward() 
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.1)               
                optimizer.step()
                scheduler.step() 
                
                if valid_emb is not None and i % 200 == 0 and i > 1000:
                    self.model.eval()
                    ws,_ = self.model(valid_emb[0])
                    ws = ws.detach().cpu().numpy()
                    ws_pre,_ = self.model(valid_emb[1]) #pre emb
                    ws_pre = ws_pre.detach().cpu().numpy()
                    ws_alpha = ws / (self.alpha * ws + 1-self.alpha)            
                    ws_pre_alpha = ws_pre / (self.alpha * ws_pre + 1-self.alpha)                        
                    val_loss = 0.5 * self.alpha * ((ws_alpha)**2).mean() + (1-self.alpha) *0.5 * ((ws_pre_alpha)**2).mean() - (ws_alpha).mean() + 1.0 * ((ws_pre_alpha.mean()-1.0)**2)                  
                    early_stop(val_loss, self.model)
                    if early_stop.early_stop:
                        logger.info("Early stopping")
                        break

                if i % 200 == 0:   
                    logger.info(f'At Iteration {i}. ' + 'Training loss = '+ str(loss.item()) + '.')                                                                                                                                        
                    logger.info('Mean of selection factors for pre-sel. sequences: ' + str(weights_pre1.mean().item()) )

        if save_checkpoint is not None and valid_emb is None:
            torch.save(self.model.state_dict(), save_checkpoint)                    
            logger.info(f'At Iteration {i} save the model to {save_checkpoint}')

    def train(self,iters,seqs_post,seqs_pre=None,batch_size=1000,save_checkpoint=None,valid_ratio=0.0,patience=5):
        #assert emb_model_path is not None, logger.info('You need to specify the directory of embedding model')
        if seqs_pre is None:
            n = len(seqs_post)            
            logger.info(f'Begin generating {n} pre-selection sequences using generation model from {self.default_gen_model.model_folder}.')
            seqs_pre = self.default_gen_model.sample(n)
            logger.info('Done!')

        if type(seqs_pre[0][0]) == str or type(seqs_pre[0][0]) == np.str or type(seqs_pre[0][0]) == np.str_: #need to get full TCR-beta
            seqs_pre_full = cdr2full(seqs_pre,multi_process=True)  #v-j-cdr3
            if type(seqs_pre_full[0]) != str:
                seqs_pre_full = [c.decode('utf-8') for c in seqs_pre_full]
            seqs_for_emb = [[s[0] for s in seqs_pre],seqs_pre_full]
            pre_emb = get_embedded_data(seqs_for_emb,self.emb_model_path)
        else :
            pre_emb = seqs_pre 

        if type(seqs_post[0][0]) == str:
            seqs_post_full = cdr2full(seqs_post,multi_process=True)  #v-j-cdr3
            if type(seqs_post_full[0]) != str:
                seqs_post_full = [c.decode('utf-8') for c in seqs_post_full]
            seqs_post = [[s[0] for s in seqs_post],seqs_post_full]
            post_emb = get_embedded_data(seqs_post,self.emb_model_path)
        else:
            post_emb = seqs_post
        loader_train = Loader(pre_emb,post_emb,batch_size)
        
        emb_valid= None
        if valid_ratio > 0:
            index = np.random.permutation(len(post_emb))            
            post_emb_val = post_emb[index[:int(len(index) * valid_ratio)]]        
            post_emb = post_emb[index[int(len(index) * valid_ratio):]]                        
            emb_valid = [post_emb_val,pre_emb]

        self.fit(iters,loader_train,save_checkpoint=save_checkpoint,valid_emb = emb_valid,patience=patience)
        
        return seqs_pre,pre_emb,post_emb
                
    def predict_weights(self,samples,batch_size=128):
        '''
        @samples: embedding of input TCRs; size of L x d
        '''
        self.model.eval() 
        batch_num = len(samples) // batch_size +1
        weights_pre = []
        for i in range(batch_num):
            if len(samples[i*batch_size:(i+1)*batch_size]) == 0:
                continue
            samples_sub = samples[i * batch_size:(i+1)*batch_size]
            if type(samples_sub[0]) == str or type(samples_sub[0]) == np.str or type(samples_sub[0]) == np.str_:
                samples_sub = get_embedded_data(samples_sub,self.emb_model_path)
            weights_pre_tmp,_ = self.model(torch.FloatTensor(samples_sub).to(self.device))  
            weights_pre.append(weights_pre_tmp.detach().cpu().numpy()[:,0])
        ws_pre = np.concatenate(weights_pre)          
        ws_pre[ws_pre > 1000] = 1000 #max  
        ws_pre[ws_pre == 0] = np.random.uniform(low=1e-5,high=1e-2,size=np.sum(ws_pre==0))
        return ws_pre
    
    def sample(self,n,prob=False):
        syn_samples,sel_factors,embeddings = sampler(self.default_gen_model,self,n)
        re_dic = {'samples':syn_samples,'sel_factors':sel_factors,'embeddings':embeddings}
        if not prob:
            return re_dic
        else:
            pgen = self.default_gen_model.p_gen(syn_samples)
            pposts = pgen * sel_factors
            re_dic['pposts'] = pposts
            re_dic['pgens'] = pgen
            return re_dic
        
    def get_prob(self,samples):
        if len(samples) == 2 and len(samples[0][0]) == 3:
            sel_factors = self.predict_weights(samples[1])
        else :
            sel_factors = self.predict_weights(samples)
        pgen = self.default_gen_model.p_gen(samples)
        pposts = pgen * sel_factors
        return pgen,pposts
    
#simulation estimators
class V_select:
    def __init__(self,samples,seed=43,temp=0.1):
        #v_probs should be a list
        np.random.seed(seed)
        #first infer the v_probs; 
        v_probs = defaultdict(int)
        for s in samples:
            v_probs[s[1]] += 1
        for key in v_probs.keys():
            v_probs[key] /= len(samples)
        self.temp = temp
        v_sel = np.random.uniform(size=len(v_probs))
        v_sel = np.exp(v_sel / temp)
        v_sel = v_sel / np.sum(v_sel)

        sum_ = 0
        self.v_probs = v_probs
        self.v_genes = list(v_probs.keys())
        probs = [v_probs[v] for v in self.v_genes]
        for i,prob in enumerate(probs):
            sum_ += prob * v_sel[i]
        self.Z = 1 / sum_
        self.v_sel = self.Z * v_sel
        self.gene2idx = {self.v_genes[i]:i for i in range(len(self.v_genes))}
    
    def predict_weights(self,samples):
        v_genes = [sample[1] for sample in samples]
        v_gene_idx = [self.gene2idx[g] for g in v_genes]
        return [self.v_sel[idx] for idx in v_gene_idx]

    def selection_model(self):
        return {self.v_genes[i]:self.v_sel[i] for i in range(len(self.v_genes))}
    
class J_select:
    def __init__(self,samples,seed=43,temp=0.1):
        #v_probs should be a list
        np.random.seed(seed)
        #first infer the v_probs; 
        j_probs = defaultdict(int)
        for s in samples:
            j_probs[s[2]] += 1
        for key in j_probs.keys():
            j_probs[key] /= len(samples)
        
        j_sel = np.random.uniform(size=len(j_probs))
        j_sel = np.exp(j_sel / temp)
        j_sel = j_sel / np.sum(j_sel)
        sum_ = 0
        self.j_probs = j_probs
        self.j_genes = list(j_probs.keys())
        probs = [j_probs[j] for j in self.j_genes]
        for i,prob in enumerate(probs):
            sum_ += prob * j_sel[i]
        self.Z = 1 / sum_
        self.j_sel = self.Z * j_sel
        self.gene2idx = {self.j_genes[i]:i for i in range(len(self.j_genes))}
    
    def predict_weights(self,samples):
        j_genes = [sample[2] for sample in samples]
        j_gene_idx = [self.gene2idx[g] for g in j_genes]
        return [self.j_sel[idx] for idx in j_gene_idx]

    def selection_model(self):
        return {self.j_genes[i]:self.j_sel[i] for i in range(len(self.j_genes))}

class VJ_select:
    def __init__(self,samples,seed=0,temp=1.0):
        #v_probs should be a list
        np.random.seed(seed)
        j_names = set()
        v_names = set()
        for s in samples:
            j_names.add(s[2])
            v_names.add(s[1])
        #infer VJ probs
        vj_probs = defaultdict(int)
        for s in samples:
            vj_probs[(s[1],s[2])] += 1
        for key in vj_probs.keys():
            vj_probs[key] /= len(samples)
        
        vj_sel = np.random.uniform(size=len(vj_probs))
        vj_sel = np.exp(vj_sel/temp)
        vj_sel = vj_sel / np.sum(vj_sel)
        sum_ = 0
        self.vj_probs = vj_probs
        self.vj_genes = list(vj_probs.keys())
        probs = [vj_probs[vj] for vj in self.vj_genes]
        for i,prob in enumerate(probs):
            sum_ += prob * vj_sel[i]
        self.Z = 1 / sum_
        self.vj_sel = self.Z * vj_sel
        self.gene2idx = {self.vj_genes[i]:i for i in range(len(self.vj_genes))}
    
    def predict_weights(self,samples):
        vj_genes = [(sample[1],sample[2]) for sample in samples]
        vj_gene_idx = [self.gene2idx[g] for g in vj_genes]
        return [self.vj_sel[idx] for idx in vj_gene_idx]

    def selection_model(self):
        return {self.vj_genes[i]:self.vj_sel[i] for i in range(len(self.vj_genes))}

class Motif_select:
    def __init__(self,samples,start_pos=5,k=3,
                 column_idx = 0,sel_factor=2,sel_pos=0,l_max=20,thre=0.005):
        #this is for cdr3        
        self.column_idx = column_idx
        #first align
        cdr3s = [s[column_idx] for s in samples]
        self.l_max = l_max
        _,cdr3s_align,_ = check_align_cdr3s(cdr3s,lmaxtrain=l_max)        
        # cdr3s_align = cdr3s
        self.start_pos = start_pos
        self.k = k
        motifs = defaultdict(int)
        total_ = 0
        for s in cdr3s_align:
            seg = s[start_pos:start_pos+k]
            if '-' in seg:
                continue
            motifs[seg] += 1
            total_ += 1
        motif_list = list(motifs.keys())
        motif_list.sort(key=lambda x: - motifs[x])
        self.motif_list = motif_list        
        
        for key in motifs.keys():
            motifs[key] /= total_
        # assert total_ > len(cdr3s_align) // 2, 'too many pad tokens; try to use different start_pos or k'        
        # print(motifs)
        self.motifs = motifs     
        self.key_fre = [(k,motifs[k]) for k in motif_list]   
        #print(self.key_fre[:10])     
        self.motifs_sel = [m for m in self.motifs if motifs[m] > thre]
        #print(len(self.motifs_sel))
        if sel_pos == 1:
            sel_pos = len(self.motifs_sel) // 2
        elif sel_pos == 2:
            sel_pos = len(self.motifs_sel) - 1
        print(sel_pos)
        motif_sel = np.ones(len(motifs.keys()))
        # motif_sel[0] = 0.1
        # motif_sel[1] = 2
        motif_sel[sel_pos] = sel_factor
        #different level
        motif_sel = np.exp(motif_sel)
        
        motif_sel = motif_sel / np.sum(motif_sel)
        sum_ = 0        
        # self.motif_list = list(motifs.keys())
        probs = [motifs[v] for v in self.motif_list]
        #print(len(probs))
        for i,prob in enumerate(probs):
            sum_ += prob * motif_sel[i]
        self.Z = 1 / sum_
        self.motif_sel = self.Z * motif_sel
        self.motif2idx = {self.motif_list[i]:i for i in range(len(self.motif_list))}
        self.param_dict = {
            'sel_factor':sel_factor,
            'start_pos':start_pos,
            'k':k,
            'sel_pos':sel_pos,            
            'column_idx':column_idx
        }     

    def predict_weights(self,samples):
        # assert len(samples[0][self.column_idx]) < 50        
        cdrs = [sample[self.column_idx] for sample in samples]        
        _,cdrs_align,_ = check_align_cdr3s(cdrs,self.l_max)
        # cdrs_align = cdrs
        weights = []       
        for cdr3 in cdrs_align:
            seg = cdr3[self.start_pos:self.start_pos + self.k]            
            seg_idx = -1 if seg not in self.motif2idx else self.motif2idx[seg]
            weights.append(0 if seg_idx == -1 else self.motif_sel[seg_idx])
        
        return weights        
    
class NN(nn.Module):
    def __init__(self,in_dims,hid_dim,out_dim=1,dropout=0.1,simulation=True):
        super().__init__()
        self.in_dims = in_dims
        LN = not simulation
        if LN:
            self.projects = nn.ModuleList([nn.Sequential(
                nn.Linear(in_dims[i], hid_dim), 
                nn.LayerNorm(hid_dim),
                nn.ReLU(),
                nn.Linear(hid_dim, hid_dim),
                nn.LayerNorm(hid_dim),
                nn.ReLU()
            ) for i in range(len(in_dims))] )
            self.project3 = nn.Sequential(
                nn.Linear(hid_dim * len(in_dims), len(in_dims) * hid_dim // 2), 
                nn.LayerNorm(len(in_dims) * hid_dim // 2),
                nn.Dropout(dropout),
                nn.ReLU(),                
                nn.Linear(len(in_dims) * hid_dim // 2, hid_dim ),                
                nn.LayerNorm(hid_dim),
                nn.Dropout(dropout),
                nn.ReLU(),                
                nn.Linear(hid_dim, out_dim),nn.ReLU())
        else :
            self.projects = nn.ModuleList([nn.Sequential(
                nn.Linear(in_dims[i], hid_dim),                 
                nn.ReLU(),
                nn.Linear(hid_dim, hid_dim),
                nn.ReLU()
            ) for i in range(len(in_dims))] )
            self.project3 = nn.Sequential(
                nn.Linear(hid_dim * len(in_dims), len(in_dims) * hid_dim // 2), 
                nn.Dropout(dropout),
                nn.ReLU(),                
                nn.Linear(len(in_dims) * hid_dim // 2, hid_dim ), 
                nn.Dropout(dropout),
                nn.ReLU(),                
                nn.Linear(hid_dim, out_dim),nn.ReLU())

    def forward(self, x):  
        xs = []
        idx = 0
        for i in range(len(self.in_dims)):
            xs.append(x[:,idx:idx+self.in_dims[i]])
            idx += self.in_dims[i]
        xs = [self.projects[i](xs[i]) for i in range(len(xs))]
        x = torch.cat(xs,-1)            
        x1  =self.project3(x)                   
        return x1,1     