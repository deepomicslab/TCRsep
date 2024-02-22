## TCRsep: T-cell receptor selection estimation procedure
TCRsep is a python software for the inference of the selection factor for immune receptor repertoires. It takes a productive TCR repertoire and pre-selection repertoire (optional) as inputs for model training. After that, it outputs the selection factors for any given TCR clonetypes (defined in the __CDR3-V-J__ format). It also outputs their post-selection probabilities that can be further utilized to analyze the receptor sharing pattern and identify indicative disease-associated receptors. 
 <br />

<img src="https://github.com/jiangdada1221/TCRsep/blob/main/figs/workflow_github.png" width="800"> <br />

## Installation
TCRsep is available on PyPI and can be installed via pip: <br />
 ```pip install tcrsep``` <br />
TCRsep depends on multiple packages. Make sure that the following dependencies are installed correctly:
 ```
torch >= 1.5.0 (Tested on torch 1.8.0+cuda11.1)
[olga](https://github.com/statbiophys/OLGA)
 ```

## Data

 All the data used in the paper is publicly available, so we suggest readers refer to the original papers for more details. We also upload the processed data which can be downloaded via [this link](https://drive.google.com/file/d/1rqgn6G2js85QS6K7mvMwOEepm4ARi54H/view?usp=sharing)

## Usage instructions

Define and train TCRpeg model:
```python
from tcrpeg.TCRpeg import TCRpeg
model = TCRpeg(embedding_path='tcrpeg/data/embedding_32.txt',load_data=True, path_train=tcrs) 
#'embedding_32.txt' records the numerical embeddings for each AA; We provide it under the 'tcrpeg/data/' folder.
#'tcrs' is the TCR repertoire ([tcr1,tcr2,....])
model.create_model() #initialize the TCRpeg model
model.train_tcrpeg(epochs=20, batch_size= 32, lr=1e-3) 
#defining and training of TCRpeg_vj can be found in tutorial.ipynb
```
Load the default models
```pyton
model = TCRpeg(embedding_path='tcrpeg/data/embedding_32.txt',load_data=False)
model.create_model(load=True,path='tcrpeg/models/tcrpeg.pth')
#TCRpeg_vj model
model_vj = TCRpeg(embedding_path='tcrpeg/data/embedding_32.txt',load_data=False,vj=True)
model_vj.create_model(vj=True,load=True,path='tcrpeg/models/tcrpeg_vj.pth')
```
Use the pretrained TCRpeg model for downstream applications:
```python
log_probs = model.sampling_tcrpeg_batch(tcrs)   #probability inference
new_tcrs = model.generate_tcrpeg(num_to_gen=1000, batch_size= 100)    #generation
embs = model.get_embedding(tcrs)    #embeddings for tcrs
```
#### Updates
The downstream applications can be also applied to CDR3+V+J data
```python
new_clonetypes = model.generate_tcrpeg_vj(num_to_gen=1000, batch_size= 100) #generation
log_probs_clonetypes = model.sampling_tcrpeg_batch(clone_types) # get the probs of CDR3_V_J
#size of clone_types: 3xlength ([[cdr1,cdr2,cdr3...],[v1,v2,v3..],[j1,j2,j3...]])
```

 We provide a tutorial jupyter notebook named [tutorial.ipynb](https://github.com/jiangdada1221/TCRpeg/blob/main/tutorial.ipynb). It contains most of the functional usages of TCRpeg which mainly consist of three parts: probability inference, numerical encodings & downstream classification, and generation. <br />

 ## Command line usages

 We have provided the scripts for the experiments in the paper via the folder [tcrpeg/scripts](https://github.com/jiangdada1221/TCRpeg/tree/main/tcrpeg/scripts). <br />

 ```
python train.py --path_train ../data/TCRs_train.csv --epoch 20 --learning_rate 0.0001 --store_path ../results/model.pth 
```
To train a TCRpeg (with vj) model, the data file needs to have the columns named 'seq', 'v', 'j'. Insert 'python train.py --h' for more details.<br />
```
python evaluate.py --test_path ../data/pdf_test.csv --model_path ../results/model.pth
```
To compute the Pearson correlation coefficient of the probability inference task on test set. <br />
```
python generate.py --model_path ../results/model.pth --n 10000 --store_path ../results/gen_seq.txt
```
Use the pretrained TCRpeg to generate new sequences. Type 'python generate.py --h' for more details <br />
```
python classify.py --path_train ../data/train.csv --path_test ../data/test.csv --epoch 20 --learning_rate 0.0001
```
Use TCRpeg-c for classification task. The files should have two columns: 'seq' and 'label'. Type 'python classify.py --h' for more details. <br /> 
Note that the parameters unspecified will use the default ones (e.g. batch size) <br /><br />
The python files and their usages are shown below: <br />

| Module name                                    | Usage                                              |    
|------------------------------------------------|----------------------------------------------------|
| TCRpeg.py                                      | Contain most functions of TCRpeg                   |
| evaluate.py                                    | Evaluate the performance of probability inference  |
| word2vec.py                                    | word2vec model for obtaining embeddings of AAs     |
| model.py                                       | Deep learning models of TCRpeg,TCRpeg-c,TCRpeg_vj  |
| classification.py                              | Apply TCRpeg-c for classification tasks            |
| utils.py                                       | N/A (contains util functions)                      |
| process_data.py                                | Construct the universal TCR pool                   |

## Contact
```
Name: Yuepeng Jiang
Email: yuepjiang3-c@my.cityu.edu.hk/yuj009@eng.ucsd.edu/jiangdada12344321@gmail.com
Note: For instant query, feel free to send me an email since I check email often. Otherwise, you may open an issue section in this repository.
```

## License

Free use of TCRpeg is granted under the terms of the GNU General Public License version 3 (GPLv3).

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
