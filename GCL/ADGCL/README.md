# Training based on AD-GCL

Our implementation is based on [AD-GCL](https://github.com/susheels/adgcl). The main modifications are in the file `test_minmax_tu.py` and `unsupervised/learning/ginfominmax.py`.

## Dependencies

Code developed and tested in Python 3.8.8 using PyTorch 1.8. Please refer to their official websites for installation and setup.

Some major requirements are given below

```shell
numpy~=1.20.1
networkx~=2.5.1
torch~=1.8.1
tqdm~=4.60.0
scikit-learn~=0.24.1
pandas~=1.2.4
gensim~=4.0.1
scipy~=1.6.2
ogb~=1.3.1
matplotlib~=3.4.2
torch-cluster~=1.5.9
torch-geometric~=1.7.0
torch-scatter~=2.0.6
torch-sparse~=0.6.9
torch-spline-conv~=1.2.1
rdkit~=2021.03.1
```

## Usage

```shell
python test_minmax_tu.py --gpu_id=0 --dataset=MUTAG --loss=info --epochs=100 --model_lr=0.001 --reg_lambda=10.0
```

or you can run the bash

```shell
bash run_adgcl.sh $gpu_id
```
