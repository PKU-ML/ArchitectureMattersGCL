# Training based on GCA

Our implementation is based on [GCA](https://github.com/CRIPAC-DIG/GCA). The main modifications are in the file `main.py` and `pGRACE/model.py`.

## Dependencies

- torch 1.4.0
- torch-geometric 1.5.0
- sklearn 0.21.3
- numpy 1.18.1

## Usage

```shell
python train.py --gpu_id=2 --dataset=Cora --encoder=GCN --loss=info --drop_scheme=degree --lr=0.001 --num_epochs=200
```

or you can run the bash

```shell
bash run_gca.sh $gpu_id
```
