DEVICE=$1

python train.py --gpu_id=$DEVICE --dataset=Cora --num_epochs=0 --act=prelu --drop_scheme=degree --num_hidden=256 --num_proj_hidden=128
python train.py --gpu_id=$DEVICE --dataset=Cora --lr=0.01 --tau=0.3 --num_epochs=1000 --act=prelu --loss=info --drop_scheme=pr --num_hidden=256 --num_proj_hidden=128 --weight_init=0.05 --iters=10 --epoch_start=400
python train.py --gpu_id=$DEVICE --dataset=Cora --lr=0.001 --tau=0.3 --num_epochs=1000 --act=prelu --loss=uniform --drop_scheme=degree --num_hidden=256 --num_proj_hidden=128 --weight_init=0.05 --iters=10 --epoch_start=400
python train.py --gpu_id=$DEVICE --dataset=Cora --lr=0.01 --tau=1.0 --num_epochs=100 --act=prelu --loss=align --drop_scheme=degree --num_hidden=256 --num_proj_hidden=128 --weight_init=0.05 --iters=10 --epoch_start=400
python train.py --gpu_id=$DEVICE --dataset=Cora --lr=0.01 --tau=0.3 --num_epochs=200 --act=prelu --loss=align --drop_scheme=pr --num_hidden=128 --num_proj_hidden=128 --use_norm --norm_type=cn --scale=20 --weight_init=0.05 --iters=10 --epoch_start=400

python train.py --gpu_id=$DEVICE --dataset=CiteSeer --num_epochs=0 --act=prelu --drop_scheme=degree --num_hidden=128 --num_proj_hidden=128
python train.py --gpu_id=$DEVICE --dataset=CiteSeer --lr=0.01 --tau=0.2 --num_epochs=100 --act=prelu --loss=info --drop_scheme=pr --num_hidden=128 --num_proj_hidden=128 --weight_init=0.05 --iters=10 --epoch_start=400
python train.py --gpu_id=$DEVICE --dataset=CiteSeer --lr=0.0001 --tau=0.2 --num_epochs=100 --act=rrelu --loss=uniform --drop_scheme=degree --num_hidden=256 --num_proj_hidden=256 --weight_init=0.05 --iters=10 --epoch_start=400
python train.py --gpu_id=$DEVICE --dataset=CiteSeer --lr=0.001 --tau=1.0 --num_epochs=200 --act=prelu --loss=align --drop_scheme=degree --num_hidden=128 --num_proj_hidden=128 --weight_init=0.05 --iters=10 --epoch_start=400
python train.py --gpu_id=$DEVICE --dataset=CiteSeer --lr=0.001 --tau=0.3 --num_epochs=200 --act=prelu --loss=align --drop_scheme=pr --num_hidden=128 --num_proj_hidden=128 --use_norm --norm_type=cn --scale=10 --weight_init=0.05 --iters=10 --epoch_start=400

python train.py --gpu_id=$DEVICE --dataset=PubMed --num_epochs=0 --act=prelu --drop_scheme=pr --num_hidden=256 --num_proj_hidden=128
python train.py --gpu_id=$DEVICE --dataset=PubMed --lr=0.0001 --tau=0.2 --num_epochs=2000 --act=prelu --loss=info --drop_scheme=degree --num_hidden=256 --num_proj_hidden=128 --weight_init=0.05 --iters=10 --epoch_start=400
python train.py --gpu_id=$DEVICE --dataset=PubMed --lr=0.0001 --tau=0.2 --num_epochs=2000 --act=rrelu --loss=uniform --drop_scheme=degree --num_hidden=256 --num_proj_hidden=256 --weight_init=0.05 --iters=10 --epoch_start=400
python train.py --gpu_id=$DEVICE --dataset=PubMed --lr=0.0001 --tau=1.0 --num_epochs=2000 --act=prelu --loss=align --drop_scheme=degree --num_hidden=128 --num_proj_hidden=128 --weight_init=0.05 --iters=10 --epoch_start=400
python train.py --gpu_id=$DEVICE --dataset=PubMed --lr=0.0001 --tau=0.1 --num_epochs=2000 --act=prelu --loss=align --drop_scheme=evc --num_hidden=256 --num_proj_hidden=128 --use_norm --norm_type=dcn --scale=50 --weight_init=0.05 --iters=10 --epoch_start=400

python train.py --gpu_id=$DEVICE --dataset=Photo --num_epochs=0 --act=prelu --drop_scheme=degree --num_hidden=256 --num_proj_hidden=128
python train.py --gpu_id=$DEVICE --dataset=Photo --lr=0.01 --tau=0.6 --num_epochs=2500 --act=prelu --loss=info --drop_scheme=pr --num_hidden=256 --num_proj_hidden=128 --weight_init=0.15 --iters=10 --epoch_start=400
python train.py --gpu_id=$DEVICE --dataset=Photo --lr=0.01 --tau=0.4 --num_epochs=2500 --act=prelu --loss=uniform --drop_scheme=degree --num_hidden=256 --num_proj_hidden=128 --weight_init=0.15 --iters=10 --epoch_start=400
python train.py --gpu_id=$DEVICE --dataset=Photo --lr=0.01 --tau=0.2 --num_epochs=2500 --act=rrelu --loss=align --drop_scheme=degree --num_hidden=128 --num_proj_hidden=128 --weight_init=0.05 --iters=10 --epoch_start=400
python train.py --gpu_id=$DEVICE --dataset=Photo --lr=0.001 --tau=0.1 --num_epochs=2000 --act=prelu --loss=align --drop_scheme=degree --num_hidden=256 --num_proj_hidden=128 --use_norm --norm_type=cn --scale=30 --weight_init=0.05 --iters=10 --epoch_start=400

python train.py --gpu_id=$DEVICE --dataset=Computers --num_epochs=0 --act=prelu --drop_scheme=pr --num_hidden=256 --num_proj_hidden=128
python train.py --gpu_id=$DEVICE --dataset=Computers --lr=0.01 --tau=0.2 --num_epochs=2000 --act=rrelu --loss=info --drop_scheme=degree --num_hidden=128 --num_proj_hidden=128 --weight_init=0.15 --iters=10 --epoch_start=400
python train.py --gpu_id=$DEVICE --dataset=Computers --lr=0.01 --tau=0.2 --num_epochs=2000 --act=rrelu --loss=uniform --drop_scheme=evc --num_hidden=128 --num_proj_hidden=128 --weight_init=0.15 --iters=10 --epoch_start=400
python train.py --gpu_id=$DEVICE --dataset=Computers --lr=0.01 --tau=0.2 --num_epochs=2000 --act=rrelu --loss=align --drop_scheme=degree --num_hidden=128 --num_proj_hidden=128 --weight_init=0.05 --iters=10 --epoch_start=400
python train.py --gpu_id=$DEVICE --dataset=Computers --lr=0.0001 --tau=0.5 --num_epochs=2000 --act=prelu --loss=align --drop_scheme=degree --num_hidden=256 --num_proj_hidden=128 --use_norm --norm_type=dcn --scale=250 --weight_init=0.05 --iters=10 --epoch_start=400