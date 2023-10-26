# Training based on GraphCL

Our implementation is based on [GraphCL](https://github.com/Shen-Lab/GraphCL). The main modifications are in the file `gsimclr.py`.

## Dependencies

* [PyTorch Geometric](https://github.com/rusty1s/pytorch_geometric#installation)==1.6.0

Then, you need to create a directory for recoreding finetuned results to avoid errors:

``` shell
mkdir logs
```

## Usage

```shell
python gsimclr.py --gpu_id=0 --dataset=MUTAG --loss=info --num_epochs=20 --lr=0.001 --hidden_dim=32
```

or you can run the bash

```shell
bash run_graphcl.sh $gpu_id
```
