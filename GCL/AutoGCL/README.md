# Training based on Auto-GCL

Our implementation is based on [Auto-GCL](https://github.com/Somedaywilldo/AutoGCL). The main modifications are in the file `unsupervised/us_main.py`.

## Dependencies

```shell
rdkit
pytorch 1.10.0
pytorch_geometric 2.0.2
```

## Usage

```shell
cd unsupervised
python us_main.py --gpu=0 --dataset=PTC_MR --loss=info --lr=0.001 --hidden-dim=128 --tau=0.1 --epochs=30
```

or you can run the bash

```shell
cd unsupervised
bash run_autogcl.sh $gpu_id
```
