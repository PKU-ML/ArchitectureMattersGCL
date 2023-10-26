# Training based on JOAO

Our implementation is based on [JOAO](https://github.com/Shen-Lab/GraphCL_Automated). The main modifications are in the file `joaov2.py`.

## Dependencies

* [torch-geometric](https://github.com/rusty1s/pytorch_geometric) >= 1.6.0
* [ogb](https://github.com/snap-stanford/ogb) == 1.2.4

## Usage

```shell
python joaov2.py --gpu_id=0 --dataset=MUTAG --loss=info --lr=0.01 --hidden_dim=32 --num_epochs=250
```

or you can run the bash

```shell
bash run_joaov2.sh $gpu_id
```
