DEVICE=$1
python train.py --gpu_id=$DEVICE --dataset=Cora --num_epochs=0 --act=prelu --drop_scheme=evc --num_hidden=128 --num_proj_hidden=128
python train.py --gpu_id=$DEVICE --dataset=Cora --lr=0.001 --tau=0.7 --num_epochs=200 --act=prelu --loss=info --drop_scheme=evc --num_hidden=128 --num_proj_hidden=128
python train.py --gpu_id=$DEVICE --dataset=Cora --lr=0.001 --tau=0.7 --num_epochs=200 --act=prelu --loss=uniform --drop_scheme=evc --num_hidden=128 --num_proj_hidden=128
python train.py --gpu_id=$DEVICE --dataset=Cora --lr=0.001 --tau=0.7 --num_epochs=200 --act=prelu --loss=align --drop_scheme=evc --num_hidden=128 --num_proj_hidden=128
python train.py --gpu_id=$DEVICE --dataset=Cora --lr=0.01 --tau=0.3 --num_epochs=200 --act=prelu --loss=align --drop_scheme=evc --num_hidden=256 --num_proj_hidden=128 --use_norm --norm_type=cn --scale=10

python train.py --gpu_id=$DEVICE --dataset=CiteSeer --num_epochs=0 --act=prelu --drop_scheme=evc --num_hidden=128 --num_proj_hidden=128
python train.py --gpu_id=$DEVICE --dataset=CiteSeer --lr=0.001 --tau=0.9 --num_epochs=200 --act=prelu --loss=info --drop_scheme=evc --num_hidden=128 --num_proj_hidden=128
python train.py --gpu_id=$DEVICE --dataset=CiteSeer --lr=0.001 --tau=0.9 --num_epochs=200 --act=prelu --loss=uniform --drop_scheme=evc --num_hidden=128 --num_proj_hidden=128
python train.py --gpu_id=$DEVICE --dataset=CiteSeer --lr=0.001 --tau=0.7 --num_epochs=200 --act=prelu --loss=align --drop_scheme=evc --num_hidden=128 --num_proj_hidden=128
python train.py --gpu_id=$DEVICE --dataset=CiteSeer --lr=0.01 --tau=0.5 --num_epochs=200 --act=prelu --loss=align --drop_scheme=pr --num_hidden=128 --num_proj_hidden=128 --use_norm --norm_type=cn --scale=10

python train.py --gpu_id=$DEVICE --dataset=PubMed --num_epochs=0 --act=prelu --drop_scheme=evc --num_hidden=128 --num_proj_hidden=128
python train.py --gpu_id=$DEVICE --dataset=PubMed --lr=0.001 --tau=0.2 --num_epochs=1500 --act=prelu --loss=info --drop_scheme=degree --num_hidden=128 --num_proj_hidden=128
python train.py --gpu_id=$DEVICE --dataset=PubMed --lr=0.001 --tau=0.9 --num_epochs=1500 --act=prelu --loss=uniform --drop_scheme=evc --num_hidden=128 --num_proj_hidden=128
python train.py --gpu_id=$DEVICE --dataset=PubMed --lr=0.001 --tau=1.0 --num_epochs=1500 --act=prelu --loss=align --drop_scheme=evc --num_hidden=128 --num_proj_hidden=128
python train.py --gpu_id=$DEVICE --dataset=PubMed --lr=0.001 --tau=0.5 --num_epochs=1000 --act=prelu --loss=align --drop_scheme=pr --num_hidden=128 --num_proj_hidden=128 --use_norm --norm_type=dcn --scale=120

python train.py --gpu_id=$DEVICE --dataset=Photo --num_epochs=0 --act=prelu --drop_scheme=pr --num_hidden=256 --num_proj_hidden=128
python train.py --gpu_id=$DEVICE --dataset=Photo --lr=0.1 --tau=0.1 --num_epochs=2000 --act=prelu --loss=info --drop_scheme=degree --num_hidden=256 --num_proj_hidden=128 --pe1=0.3 --pe2=0.5 --pf1=0.1 --pf2=0.1
python train.py --gpu_id=$DEVICE --dataset=Photo --lr=0.001 --tau=0.1 --num_epochs=2000 --act=prelu --loss=uniform --drop_scheme=pr --num_hidden=256 --num_proj_hidden=128 --pe1=0.3 --pe2=0.5 --pf1=0.1 --pf2=0.1
python train.py --gpu_id=$DEVICE --dataset=Photo --lr=0.01 --tau=0.1 --num_epochs=2000 --act=prelu --loss=align --drop_scheme=pr --num_hidden=256 --num_proj_hidden=128
python train.py --gpu_id=$DEVICE --dataset=Photo --lr=0.01 --tau=0.1 --num_epochs=1500 --act=prelu --loss=align --drop_scheme=degree --num_hidden=256 --num_proj_hidden=128 --use_norm --norm_type=cn --scale=50

python train.py --gpu_id=$DEVICE --dataset=Computers --num_epochs=0 --act=prelu --drop_scheme=degree --num_hidden=256 --num_proj_hidden=128
python train.py --gpu_id=$DEVICE --dataset=Computers --lr=0.001 --tau=0.1 --num_epochs=1500 --act=prelu --loss=info --drop_scheme=degree --num_hidden=256 --num_proj_hidden=128 --pe1=0.5 --pe2=0.5 --pf1=0.2 --pf2=0.1
python train.py --gpu_id=$DEVICE --dataset=Computers --lr=0.001 --tau=0.1 --num_epochs=2000 --act=prelu --loss=uniform --drop_scheme=degree --num_hidden=256 --num_proj_hidden=128 --pe1=0.5 --pe2=0.5 --pf1=0.2 --pf2=0.1
python train.py --gpu_id=$DEVICE --dataset=Computers --lr=0.001 --tau=0.1 --num_epochs=1500 --act=prelu --loss=align --drop_scheme=degree --num_hidden=256 --num_proj_hidden=128
python train.py --gpu_id=$DEVICE --dataset=Computers --lr=1e-5 --tau=0.3 --num_epochs=2000 --act=prelu --loss=align --drop_scheme=evc --num_hidden=256 --num_proj_hidden=128 --use_norm --norm_type=dcn --scale=30