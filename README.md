## TCRsep: T-cell receptor selection estimation procedure
TCRsep is a python software for the inference of the selection factor for immune receptor repertoires. It takes a productive TCR repertoire and pre-selection repertoire (optional) as inputs for model training. After that, it outputs the selection factors for any given TCR clonetypes (defined in the __CDR3-V-J__ format). It also outputs their post-selection probabilities that can be further utilized to analyze the receptor sharing pattern and identify indicative disease-associated receptors. 
 <br />

<img src="https://github.com/jiangdada1221/TCRsep/blob/main/figs/workflow_github.png" width="800"> <br />

## Installation
Install TCRsep via Github (recommended): <br />
```
git clone https://github.com/jiangdada1221/TCRsep.git
cd TCRsep
pip install .
``` 
TCRsep is also available on PyPI and can be installed via pip: ```pip install tcrsep``` <br /> <br />
TCRsep depends on multiple packages. It's __highly recommended__ to manually install the following packages first. If the installation fails, make sure that the following dependencies are installed correctly: <br /> 
`torch` >= 1.8.0 (Tested on torch [1.8.0+cuda11.1](https://pytorch.org/get-started/previous-versions/#v180))<br />
[`olga`](https://github.com/statbiophys/OLGA) (For modeling the generation of TCR) <br />
[`tcr2vec`](https://github.com/jiangdada1221/TCR2vec) (For embedding TCR) <br /> 

## Usage instructions
#### 1. Train a TCRsep model: 
type `python train.py -h` to display all the commandline options: 
|Commands|Description|
|--|--|
|`-h, --help`|show the help message and exit|
|`--post_data_path=FILE`|(__required__) The path to the input repertoire file (or its embedding file). An example: `data/example.csv` (`data/example_emb.npy.gz`). If given the embedding path, will save the time to embed input TCRs.| 
|`--save_dir=DIR`|The directory that saves all the output files. Default `result/`.|  
|`--pre_data_path=FILE`|The path to the pre-selection repertoire file (or embedding). If not specified, pre-selection TCRs will be automatically generated by the generation model incorporated in TCRsep.|
|`--gen_model_path=DIR`|The path to the generation model. If not specified, will use the default generation model inferred on Emerson data.|
|`--epochs=NUM`|Iterations for the training process. Default `500`.|
|`--alpha=NUM`|The parameter α. Recommended using the default value `0.1`.|                     
|`--batchsize=NUM`|Batch size. Default `512`.|
|`--val_ratio=NUM`|Fraction of the data serving as validation. Default `0.1`.|
|`--lr=NUM`|Learning rate. Default `1e-4`.|  
|`--device=STR`|The device to run TCRsep. Default `cuda:0`.|
|`--save_emb`|If specified, will save the embedding of the input sequences. Default `False`.|

__Notes:__ the data file (`.csv/.tsv`) needs at least three columns specifying the CDR3β amino acid sequences, V genes, and J genes: `CDR3.beta`, `V` and `J`. The `save_dir` will contain the pre-selection repertoire file, embeddings of pre- and post-selection repertoires, the selection model, and a json file recording the input arguments.  

#### 2. Use the TCRsep to infer selection factors, pre- and post-selection probabilities:
type `python eval.py -h` to display all the commandline options:
|Commands|Description|
|--|--|
|`--data_path=FILE`|(__required__) The path to the query repertoire file that needs to be evaluated. An example: `data/query_data.csv`.| 
|`--data_emb_path=FILE`|The path to the embedding of the query repertoire. If specified, will save the time for embedding.|  
|`--sel_model_path=DIR`|The path to the directory of the TCRsep model. If not specified, will use the default model inferred on Emerson data.|  
|`--save_dir=DIR`|The directory that saves all the output files. Default `result_eval/`.|  
|`--device=STR`|The device to run TCRsep. Default `cuda:0`.|

#### 3. Usages of TCRsep in Python script
3.1 Utilities of TCRsep module:
```python
from tcrsep.estimator import TCRsep
sel_model = TCRsep(default_sel_model=True)
query_tcrs = [['CASSLGAGGSGTEAFF','TRBV7-9','TRBJ1-1'], ['CASTKAGGSSYEQYF','TRBV6-5','TRBJ2-7']]
sel_factors = sel_model.predict_weights(query_tcrs) #obtain selection factors
pgens, pposts = sel_model.get_prob(query_tcrs) #obtain pre- and post-selection probs 

# draw samples from p_post
post_samples = sel_model.sample(N=10)
```
3.2 Sharing analysis by TCRsep:
```python
from tcrsep.sharing_analysis import Sharing, DATCR
sharing_predictor = Sharing('data/sharing') 

# predict sharing numbers of TCRs in query_data.csv among reps in the folder, "data/sharing"
sharing_pre,sharing_real = sharing_predictor.predict_sharing('data/query_data_evaled.csv') 

# predict the sharing spectrum for reps in "data/sharing"
spectrum_pre,spectrum_real = sharing_predictor.sharing_spectrum(est_num=100000) 

# identify DATCRs
DATCR_predictor = DATCR('data/sharing')
pvalues = DATCR_predictor.pvalue('data/query_data_evaled.csv')
```
__Notes:__ the `query_data_evaled.csv` should contain an additional column `ppost` specifying the post-selection probabilities. Please [eval](https://github.com/jiangdada1221/TCRsep?tab=readme-ov-file#2-use-the-tcrsep-to-infer-selection-factors-pre--and-post-selection-probabilities) the `query_data` file first if it only contains the `CDR3-V-J` information.  

#### 4. A quick demo of training a TCRsep model using Python script
```python
import pandas as pd
from tcrsep.estimator import TCRsep
#Load data
post_df = pd.read_csv('data/demo/post_seqs.csv')
gen_df = pd.read_csv('data/demo/gen_seqs.csv')
post_samples = post_df[['CDR3.beta','V','J']].values
gen_samples = gen_df[['CDR3.beta','V','J']].values

#Infer the selection model
sel_model = TCRsep()
sel_model.train(50,post_samples,gen_samples,valid_ratio=0.0)

#Load sequence data and predict selection factors
predictions = sel_model.predict_weights(post_samples)

#Save the TCRsep model
sel_model.save(save_dir='Directory_path')

#Load the TCRsep model
from tcrsep.estimator import load_tcrsep
sel_model = load_tcrsep(load_dir='Directory_path')
```

For other customized usages, we provide the following module descriptions. Users can refer to the docstrings included in the script files for more details.

| Script name                                    | Usage                                              |    
|------------------------------------------------|----------------------------------------------------|
| estimator.py                              | The TCRsep module                                  |
| train.py                                  | Train TCRsep by command line                       |
| eval.py                                   | Inference of sel_factors and probs by command line |
| pgen.py                                   | The generation module that relies on the OLGA package|
| sharing_analysis.py                       | Modules used for sharing analysis                      |
| utils.py                                  | N/A (contains utility functions)                  |

## Processed data
All the data used in the manuscript is publicly available, so we suggest readers refer to the original papers for more details. We also provide our processed data which can be publicly [downloaded](https://drive.google.com/file/d/1tQTeZn-tbKcHH5_NsArb5JAhCVcUmOy8/view?usp=sharing).

__Data details__:
- 📁 `data/` 
  - 📁 `Chu_reps/`
  - 📁 `CMV_reps/`
    - 📁 `HIP_batch/`
    - 📁 `Keck_batch/`
  - 📁 `ImmuneCODE_reps/`
  - 📁 `Simulation/`
    - 📁 `simulation_reps/`
    - `Simulation_info.csv` Record the 500 selected pairs and their corresponding JSD values
  - 📁 `Specific_TCRs/`
    - `vdjdb_CMV_control.csv` CMV specific TCRs and control TCRs
  - 📁 `Validation_cohorts/` 
    - 📁 `RA_reps`
    - 📁 `T1D_reps`
    - 📁 `CMV_reps`
  - 📁 `Pretraining/`
    - `train.csv`
    - `valid.csv`

Note that some files were saved in the compressed format, requiring the argument `compression='gzip'` to load.

## Contact
```
Author: Yuepeng Jiang
Email: yuepjiang3-c@my.cityu.edu.hk/yuj009@eng.ucsd.edu/jiangdada12344321@gmail.com
Note: For instant query, feel free to send me an email since I check email often. 
Otherwise, you may open an issue section in this repository.
```

## License

Free use of TCRsep is granted under the terms of the MIT License.

<!-- ## Citation 
```

``` -->
