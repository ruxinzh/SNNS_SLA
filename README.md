# ADVANCING-SINGLE-SNAPSHOT-DOA-ESTIMATION-WITH-SIAMESE-NEURAL-NETWORKS-FOR-SPARSE-LINEAR-ARRAYS
This is the code for paper "Advancing single-snapshot DOA estimation with siamese neural networks for sparse linear arrays" 

## Simulated dataset generation for trianing and validation 
``` sh
python scr/dataset_gen.py --output_dir './' --num_samples_val 10 --num_samples_train 50 --N 10 --max_targets 3 
```

## Encoder & Network architectures 

<p align="center">
  <img src="https://github.com/ruxinzh/SNNS_SLA/blob/main/fig/Encoder.PNG" width="580" height="340">
</p>

<p align="center">
  <img src="https://github.com/ruxinzh/SNNS_SLA/blob/main/fig/Network.PNG" width="765" height="296">
</p>
## Training 
BaseNet1: without SA layer and contrasive loss
```sh
python train.py --use_sparse False --use_siamese False
```
BaseNet2: with SA layer and without contrasive loss
``` sh
python train.py --use_sparse True --use_siamese False
```
Proposed Network: with SA layer and contrasive loss
``` sh
python train.py --use_sparse True --use_siamese True
```
## Evaluation 

### Feature Analysis
For ULA
``` sh
python feature_analysis.py --sparsity 0
```
For SLA
``` sh
python feature_analysis.py --sparsity 0.3
```
Expected outputs:
<p align="center">
  <img src="https://github.com/ruxinzh/SNNS_SLA/blob/main/fig/featureAnalysis.PNG" width="600" height="600">
</p>

### DOA Estimation
``` sh
python eval.py --num_simulations 5000 --threshold 0.5 --num_targets 3
```
Expected outputs:
<p align="center">
  <img src="https://github.com/ruxinzh/SNNS_SLA/blob/main/fig/accuracy.png" width="385" height="320">
  <img src="https://github.com/ruxinzh/SNNS_SLA/blob/main/fig/precision.png" width="385" height="320">
</p>

<p align="center">
  <img src="https://github.com/ruxinzh/SNNS_SLA/blob/main/fig/F1.png" width="385" height="320">
  <img src="https://github.com/ruxinzh/SNNS_SLA/blob/main/fig/recall.png" width="385" height="320">
</p>


## Enviroment 
The Conda environment required for this project is specified in the file 'conda_env.txt'. This file contains a list of all the necessary Python packages and their versions to ensure compatibility and reproducibility of the project's code.






