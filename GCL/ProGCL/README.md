# Training based on ProGCL

Our implementation is based on [ProGCL: Rethinking Hard Negative Mining in Graph Contrastive Learning](https://github.com/junxia97/ProGCL). The main modifications are in the file `main.py` and `pGRACE/model.py`.

## Requirements

* Python 3.7.4
* PyTorch 1.7.0
* torch_geometric 1.5.0
* tqdm
  
## Usage

```shell
python train.py --gpu_id=0 --dataset=Cora --lr=0.001 --num_epochs=200 --act=prelu --loss=info --weight_init=0.05 --iters=10 --epoch_start=400
```

or you can run the bash

```shell
bash run_progcl.sh $gpu_id
```
