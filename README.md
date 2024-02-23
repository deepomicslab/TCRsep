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
Train a TCRsep model: type `python train.py -h` to display all the commandline options: 
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

Use the TCRsep to infer selection factors, pre- and post-selection probabilities: type `python eval.py -h` to display all the commandline options:
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


```python
from tcrsep.estimator import TCRsep
sel_model = TCRsep() 
sel_model.train(iters=100,seqs_post=clonetypes)
#'embedding_32.txt' records the numerical embeddings for each AA; We provide it under the 'tcrsep/data/' folder.
#'tcrs' is the TCR repertoire ([tcr1,tcr2,....])
model.create_model() #initialize the TCRsep model
model.train_tcrsep(epochs=20, batch_size= 32, lr=1e-3) 
#defining and training of TCRsep_vj can be found in tutorial.ipynb
```
Load the default models
```pyton
model = TCRsep(embedding_path='tcrsep/data/embedding_32.txt',load_data=False)
model.create_model(load=True,path='tcrsep/models/tcrsep.pth')
#TCRsep_vj model
model_vj = TCRsep(embedding_path='tcrsep/data/embedding_32.txt',load_data=False,vj=True)
model_vj.create_model(vj=True,load=True,path='tcrsep/models/tcrsep_vj.pth')
```
Use the pretrained TCRsep model for downstream applications:
```python
log_probs = model.sampling_tcrsep_batch(tcrs)   #probability inference
new_tcrs = model.generate_tcrsep(num_to_gen=1000, batch_size= 100)    #generation
embs = model.get_embedding(tcrs)    #embeddings for tcrs
```
#### Updates
The downstream applications can be also applied to CDR3+V+J data
```python
new_clonetypes = model.generate_tcrsep_vj(num_to_gen=1000, batch_size= 100) #generation
log_probs_clonetypes = model.sampling_tcrsep_batch(clone_types) # get the probs of CDR3_V_J
#size of clone_types: 3xlength ([[cdr1,cdr2,cdr3...],[v1,v2,v3..],[j1,j2,j3...]])
```

 We provide a tutorial jupyter notebook named [tutorial.ipynb](https://github.com/jiangdada1221/TCRsep/blob/main/tutorial.ipynb). It contains most of the functional usages of TCRsep which mainly consist of three parts: probability inference, numerical encodings & downstream classification, and generation. <br />

 ## Command line usages

 We have provided the scripts for the experiments in the paper via the folder [tcrsep/scripts](https://github.com/jiangdada1221/TCRsep/tree/main/tcrsep/scripts). <br />

 ```
python train.py --path_train ../data/TCRs_train.csv --epoch 20 --learning_rate 0.0001 --store_path ../results/model.pth 
```
To train a TCRsep (with vj) model, the data file needs to have the columns named 'seq', 'v', 'j'. Insert 'python train.py --h' for more details.<br />
```
python evaluate.py --test_path ../data/pdf_test.csv --model_path ../results/model.pth
```
To compute the Pearson correlation coefficient of the probability inference task on test set. <br />
```
python generate.py --model_path ../results/model.pth --n 10000 --store_path ../results/gen_seq.txt
```
Use the pretrained TCRsep to generate new sequences. Type 'python generate.py --h' for more details <br />
```
python classify.py --path_train ../data/train.csv --path_test ../data/test.csv --epoch 20 --learning_rate 0.0001
```
Use TCRsep-c for classification task. The files should have two columns: 'seq' and 'label'. Type 'python classify.py --h' for more details. <br /> 
Note that the parameters unspecified will use the default ones (e.g. batch size) <br /><br />
The python files and their usages are shown below: <br />

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
