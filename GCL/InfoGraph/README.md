# Training based on GraphCL

Our implementation is based on [InfoGraph](https://github.com/sunfanyunn/InfoGraph). The main modifications are in the file `main.py` and `losses.py`.

## Dependencies

Tested on pytorch 1.6.0 and [pytorch\_geometric](https://github.com/rusty1s/pytorch_geometric) 1.6.1

Experiments reported on the paper are conducted in 2019 with `pytorch_geometric==1.3.1`.
Note that the code regarding of QM9 dataset in pytorch\_geometric has been changed since then. Thus, if you run this repo with `pytorch_geometric>=1.6.1`, you may obtain results differ from those reported on the paper.

## Usage

```shell
python main.py --gpu_id=0 --DS=MUTAG --loss=jsd --lr=0.01 --hidden_dim=512 --num_epochs=100 --num_layers=4
```

or you can run the bash

```shell
bash run_infograph.sh $gpu_id
```
