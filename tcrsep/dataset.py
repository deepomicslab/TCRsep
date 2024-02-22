from typing import List, Sequence, Dict, Any
import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from tape import TAPETokenizer

def pad_sequences(sequences: Sequence, constant_value=0, dtype=None) -> np.ndarray:
    batch_size = len(sequences)
    shape = [batch_size] + np.max([seq.shape for seq in sequences], 0).tolist()

    if dtype is None:
        dtype = sequences[0].dtype

    if isinstance(sequences[0], np.ndarray):
        array = np.full(shape, constant_value, dtype=dtype)
    elif isinstance(sequences[0], torch.Tensor):
        array = torch.full(shape, constant_value, dtype=dtype)

    for arr, seq in zip(array, sequences):
        arrslice = tuple(slice(dim) for dim in seq.shape)
        arr[arrslice] = seq

    return array

class TCRLabeledDset(Dataset):
    '''
    The dataset module for TCR data; used for TCR2vec to embed TCR sequence
    '''
    def __init__(self,file,only_tcr=False,use_column = 'CDR3.beta'):
        if type(file) == str:
            d = pd.read_csv(file)
        else :
            d = pd.DataFrame({use_column:file})
        self.only_tcr = only_tcr
        if only_tcr:
            cs = d[use_column].values
            self.seqs = cs
        else: 
            cs,es = d[use_column].values, d['Label'].values ####
            self.seqs = cs
            self.labels = es        
        self.tokenizer = TAPETokenizer(vocab='iupac')

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self,idx):
        seq = self.seqs[idx]                   
        tokens = self.tokenizer.tokenize(seq)
        tokens = self.tokenizer.add_special_tokens(tokens)
        token_ids = np.array(
            self.tokenizer.convert_tokens_to_ids(tokens), np.int64)                
        input_mask = np.ones_like(token_ids)
        if not self.only_tcr:
            label = self.labels[idx]
            #label = self.e2l[label]
            return token_ids,input_mask,label, seq
        else :
            return token_ids, input_mask,seq

    def collate_fn(self, batch: List[Any]) -> Dict[str, torch.Tensor]:
        # input_ids, input_mask, lm_label_ids, clan, family = tuple(zip(*batch))
        if not self.only_tcr:
            input_ids, input_mask, label, seq = tuple(zip(*batch))
        else :
            input_ids, input_mask, seq = tuple(zip(*batch))
        input_ids = torch.from_numpy(pad_sequences(input_ids, 0))
        input_mask = torch.from_numpy(pad_sequences(input_mask, 0))       
        if not self.only_tcr:
            return {'input_ids': input_ids,
                'input_mask': input_mask,
                'label':list(label),
                'seq':list(seq)}
        else :
            return {'input_ids': input_ids,
                'input_mask': input_mask,
                'seq':list(seq)}

def Loader(gen_pre_emb,gen_post_emb,batch_size=1000): 
    '''
    Data loader for TCR embeddings; continiously output embeddings
    '''
    iters1 = len(gen_pre_emb) // batch_size - 1 if len(gen_pre_emb) % batch_size == 0 else len(gen_pre_emb) // batch_size
    iters2 = len(gen_post_emb) // batch_size - 1 if len(gen_post_emb) % batch_size == 0 else len(gen_post_emb) // batch_size
    index1 = 0
    index2 = 0
    permutations_pre = np.random.permutation(len(gen_pre_emb))
    permutations_post = np.random.permutation(len(gen_post_emb))

    while True:
        if index1 == iters1:
            index1 = 0
            #shuffle
            permutations_pre = np.random.permutation(len(gen_pre_emb))

        if index2 == iters2:
            index2=  0         
            permutations_post = np.random.permutation(len(gen_post_emb))

        index1 += 1
        index2 += 1        
        
        samples_emb_pre = gen_pre_emb[permutations_pre[index1 * batch_size : (index1+1)*batch_size]]        
        samples_emb_post = gen_post_emb[permutations_post[index2 * batch_size : (index2+1)*batch_size]]

        yield samples_emb_pre,samples_emb_post