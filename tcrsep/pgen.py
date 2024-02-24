import numpy as np
import pandas as pd 
import olga.load_model as load_model
import olga.generation_probability as pgen
import olga.sequence_generation as seq_gen
import multiprocessing as mp 
from tcrsep.utils import divide_chunks
from math import ceil
from collections import defaultdict
import inspect
import tcrsep

class Generation_model:
    def __init__(self,model_folder=None,processes=None,change_sep=False):
        if model_folder is None or model_folder == "None":
            package_path = inspect.getfile(tcrsep)
            model_folder = package_path.split('__init__.py')[0] + 'models/generation_model/CMV_whole'
        self.model_folder = model_folder
        params_file_name = f'{model_folder}/model_params.txt'
        marginals_file_name = f'{model_folder}/model_marginals.txt'
        V_anchor_pos_file =f'{model_folder}/V_gene_CDR3_anchors.csv'
        J_anchor_pos_file = f'{model_folder}/J_gene_CDR3_anchors.csv'
        if change_sep:
            d = pd.read_csv(V_anchor_pos_file,sep=';')
            d.to_csv(V_anchor_pos_file,sep=',',index=False)
            d = pd.read_csv(J_anchor_pos_file,sep=';')
            d.to_csv(J_anchor_pos_file,sep=',',index=False)
        genomic_data = load_model.GenomicDataVDJ()
        genomic_data.load_igor_genomic_data(params_file_name, V_anchor_pos_file, J_anchor_pos_file)
        self.genomic_data = genomic_data
        #self.genomic_data.genv[i][0]
        generative_model = load_model.GenerativeModelVDJ()
        generative_model.load_and_process_igor_model(marginals_file_name)
        self.igor = generative_model
        #igor.PV
        self.eva_model = pgen.GenerationProbabilityVDJ(generative_model, genomic_data)
        self.gen_model = seq_gen.SequenceGenerationVDJ(generative_model, genomic_data)
        self.norm = self.eva_model.compute_regex_CDR3_template_pgen('CX{0,}')
        if processes is None: self.processes = mp.cpu_count()
        else: self.processes = processes

    def estimate_gene_usage(self,num=1000000):
        samples = self.sample(num=num)
        v_gene_usage = defaultdict(int)
        j_gene_usage = defaultdict(int)
        for sample in samples:
            v_gene_usage[sample[1]] += 1.0
            j_gene_usage[sample[2]] += 1.0
        for k in v_gene_usage.keys():
            v_gene_usage[k] /= num
        for k in j_gene_usage.keys():
            j_gene_usage[k] /= num
        self.v_gene_usage = v_gene_usage
        self.j_gene_usage = j_gene_usage
        return v_gene_usage,j_gene_usage

    def estimate_f(self,num=100000):
        samples = self.sample_labels(num)
        return np.sum(samples) / len(samples)
    
    def p_gen(self,samples):
        if len(samples) > 1000:
            #print('using multi_thread')
            processes = mp.cpu_count()
            n = ceil(len(samples) / processes) 
            samples_chunks = list(divide_chunks(samples, n))
            pool = mp.Pool(processes=processes)
            res = pool.map(self._compute_pgen,samples_chunks)
            pgens = np.concatenate(res)
            pool.close()
            pool.join()
            return pgens / self.norm #already normalized
        else :
            return np.array(self._compute_pgen(samples)) / self.norm

    def _compute_pgen(self,samples):
        pgens = []
        for s in samples:            
            if len(s) == 3:                                
                pgens.append(self.eva_model.compute_aa_CDR3_pgen(s[0],s[1],s[2]))
            else:                                          
                pgens.append(self.eva_model.compute_aa_CDR3_pgen(s))
        return pgens

    def generate(self,seed=0):
        if seed != 0:
            np.random.seed(seed)
        seq=self.gen_model.gen_rnd_prod_CDR3()
        return [seq[1], self.genomic_data.genV[seq[2]][0].split('*')[0], self.genomic_data.genJ[seq[3]][0].split('*')[0],seq[0]]
        #return [seq[1], self.genomic_data.genV[seq[2]][0], self.genomic_data.genJ[seq[3]][0],seq[0]]
    
    def generate_labels(self,seed=0):
        if seed != 0:
            np.random.seed(seed)
        seq=self.gen_model.gen_label()
        return [seq]
    
    def sample_labels(self,num):
        #conserved_J_residues='ABCEDFGHIJKLMNOPQRSTUVWXYZ'            
        pool = mp.Pool(processes=self.processes)
        seeds=np.random.randint(2**32-1,size=num)
        seqs = np.array(pool.map(self.generate_labels,seeds))
        pool.close()        
        pool.join()
        return seqs

    def sample(self,num,nn_seq=False):
        #conserved_J_residues='ABCEDFGHIJKLMNOPQRSTUVWXYZ'            
        pool = mp.Pool(processes=self.processes)
        seeds=np.random.randint(2**32-1,size=num)
        seqs = np.array(pool.map(self.generate,seeds))
        pool.close()        
        pool.join()

        if not nn_seq:
            seqs = seqs[:,:-1] 
        return seqs
    
    def valid(self,seq):
        if len(seq) <=5 : 
            return False
        if len(seq) >30:
            return False ####pay attention to that
        if 's' in seq:
            return False
        return True    

