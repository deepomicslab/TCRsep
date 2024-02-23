## TCRsep: T-cell receptor selection estimation procedure
TCRsep is a python software for the inference of the selection factor for immune receptor repertoires. It takes a productive TCR repertoire and pre-selection repertoire (optional) as inputs for model training. After that, it outputs the selection factors for any given TCR clonetypes (defined in the __CDR3-V-J__ format). It also outputs their post-selection probabilities that can be further utilized to analyze the receptor sharing pattern and identify indicative disease-associated receptors. 
 <br />

<img src="https://github.com/jiangdada1221/TCRsep/blob/main/figs/workflow_github.png" width="800"> <br />

## Installation
TCRsep is available on PyPI and can be installed via pip: <br />
 ```pip install tcrsep``` <br />
TCRsep depends on multiple packages. Make sure that the following dependencies are installed correctly: <br /> [torch](https://pytorch.org/get-started/previous-versions/#v180) >= 1.5.0 (Tested on torch 1.8.0+cuda11.1)<br />
[olga](https://github.com/statbiophys/OLGA) (For modeling the generation of TCR) <br />
[tcr2vec](https://github.com/jiangdada1221/TCR2vec) (For embedding TCR) 

## Usage instructions
__1. Train a TCRsep model__: type `python train.py -h` to display all the commandline options: 
|Commands|Description|
|--|--|
|`-h, --help`|show the help message and exit|
|`--post_data_path=FILE`|(__required__) The path to the input repertoire file (or its embedding file). An example: `data/HIP13610.csv` (`data/HIP13610_emb.npy.gz`). If given the embedding path, will save the time to embed input TCRs.| 
|`--save_dir=DIR`|The directory that saves all the output files. Default `result/`.|  
|`--pre_data_path=FILE`|The path to the pre-selection repertoire file (or embedding). If not specified, pre-selection TCRs will be automatically generated by the generation model incorporated in TCRsep.|
|`--emb_model_path=DIR`|The path to the directory of the embedding model. If not specified, will use the pretrained TCR2vec and CDR3vec models.|
|`--gen_model_path=DIR`|The path to the generation model. If not specified, will use the default generation model inferred on Emerson data.|
|`--iters=NUM`|Iterations for the training process. Default 10,000.|
|`--alpha=NUM`|The parameter α. Recommended using the default value 0.1.|                       
|`--batchsize=NUM`|Batch size. Default 1,000.|
|`--val_ratio=NUM`|Fraction of the data serving as validation. Default 0.1.|  
|`--simulation`|Set to True in simulation experiments. Default False.|

__Notes:__ the data file (.csv/.tsv) needs at least three columns specifying the CDR3β amino acid sequences, V genes, and J genes: __CDR3.beta__, __V__ and __J__. The save_dir will contain the pre-selection repertoire file, embeddings of pre- and post-selection repertoires, the selection model, and a json file recording the input arguments.  

__2. Use the TCRsep to infer selection factors, pre- and post-selection probabilities__:<br />
type `python eval.py -h` to display all the commandline options:
|Commands|Description|
|--|--|
|`--data_path=FILE`|(__required__) The path to the query repertoire file that needs to be evaluated. An example: `data/query_data.csv`.| 
|`--data_emb_path=FILE`|The path to the embedding of the query repertoire. If specified, will save the time for embedding.|  
|`--sel_model_path=FILE`|The path to the TCRsep model. If not specified, will use the default model inferred on Emerson data.|  
|`--save_dir=DIR`|The directory that saves all the output files. Default `result_eval/`.|  
|`--emb_model_path=DIR`|The path to the directory of the embedding model. If not specified, will use the pretrained TCR2vec and CDR3vec models.|
|`--gen_model_path=DIR`|The path to the generation model. If not specified, will use the default generation model inferred on Emerson data.|
|`--alpha=NUM`|The parameter α of TCRsep. Default 0.1.|                       
|`--simulation`|Set to True in simulation experiments. Default False.|

__3 Usages of TCRsep in Python script__ <br />
3.1 Utilities of TCRsep module:
```python
from tcrsep.estimator import TCRsep
sel_model = TCRsep(default_sel_model=True)
query_tcrs = [['CASTQKPSYEQYF','TRBV6-9','TRBJ2-7'], ['CARGPYNEQFF','TRBV6-9','TRBJ2-1']]
sel_factors = sel_model.predict_weights(query_tcrs) #obtain selection factors
pgens, pposts = sel_model.get_prob(query_tcrs) #obtain pre- and post-selection probs 

#draw samples from p_post
post_samples = sel_model.sample(n=10)
```
3.2 Sharing analysis by TCRsep:
```python
from tcrsep.sharing_analysis import Sharing, DATCR
sharing_predictor = Sharing('data/sharing')

#predict sharing numbers of TCRs in query_data.csv among 
#repertoires in the folder data/sharing
sharing_pre,sharing_real = sharing_predictor.predict_sharing('data/query_data_evaled.csv') 

#predict the sharing spectrum
spectrum_pre,spectrum_real = sharing_predictor.sharing_spectrum(est_num=100000) 

#Identify DATCRs
DATCR_predictor = DATCR('data/sharing')
pvalues = DATCR_predictor.pvalue('data/query_data_evaled.csv')
```

| Module name                                    | Usage                                              |    
|------------------------------------------------|----------------------------------------------------|
| TCRsep.py                                      | Contain most functions of TCRsep                   |
| evaluate.py                                    | Evaluate the performance of probability inference  |
| word2vec.py                                    | word2vec model for obtaining embeddings of AAs     |
| model.py                                       | Deep learning models of TCRsep,TCRsep-c,TCRsep_vj  |
| classification.py                              | Apply TCRsep-c for classification tasks            |
| utils.py                                       | N/A (contains util functions)                      |
| process_data.py                                | Construct the universal TCR pool                   |

## Contact
```
Name: Yuepeng Jiang
Email: yuepjiang3-c@my.cityu.edu.hk/yuj009@eng.ucsd.edu/jiangdada12344321@gmail.com
Note: For instant query, feel free to send me an email since I check email often. Otherwise, you may open an issue section in this repository.
```

## License

Free use of TCRsep is granted under the terms of the GNU General Public License version 3 (GPLv3).

## Citation 
```
@article{jiang2023deep,
  title={Deep autoregressive generative models capture the intrinsics embedded in T-cell receptor repertoires},
  author={Jiang, Yuepeng and Li, Shuai Cheng},
  journal={Briefings in Bioinformatics},
  volume={24},
  number={2},
  pages={bbad038},
  year={2023},
  publisher={Oxford University Press}
}
```
