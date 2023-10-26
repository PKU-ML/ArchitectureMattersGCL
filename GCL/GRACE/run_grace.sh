DEVICE=$1
python train.py --gpu_id=$DEVICE --dataset=Cora --loss=info --encoder=GCN --tau=0.8 --act=prelu --num_hidden=128 --num_proj_hidden=128  --num_epochs=0
python train.py --gpu_id=$DEVICE --dataset=Cora --loss=info --encoder=GCN --tau=0.8 --act=prelu --num_hidden=128 --num_proj_hidden=128 --pe1=0.0 --pe2=0.1 --pf1=0.2 --pf2=0.2 --lr=0.001 --num_epochs=1000
python train.py --gpu_id=$DEVICE --dataset=Cora --loss=uniform --encoder=GCN --tau=1.0 --act=prelu --num_hidden=128 --num_proj_hidden=128 --pe1=1.0 --pe2=0.0 --pf1=0.2 --pf2=0.2 --lr=0.001 --num_epochs=1000
python train.py --gpu_id=$DEVICE --dataset=Cora --loss=align --encoder=GCN --tau=0.8 --act=prelu --num_hidden=128 --num_proj_hidden=128 --pe1=0.0 --pe2=0.1 --pf1=0.2 --pf2=0.2 --lr=0.001 --num_epochs=1000
python train.py --gpu_id=$DEVICE --dataset=Cora --loss=align --encoder=GCN --tau=0.8 --act=prelu --num_hidden=128 --num_proj_hidden=128 --pe1=0.2 --pe2=0.2 --pf1=0.2 --pf2=0.2 --lr=0.001 --num_epochs=200 --use_norm --norm_type=cn --scale=5 
python train.py --gpu_id=$DEVICE --dataset=Cora --loss=info --encoder=GCN --tau=0.6 --act=prelu --num_hidden=128 --num_proj_hidden=128 --aug_type=none --lr=0.001 --num_epochs=1000
python train.py --gpu_id=$DEVICE --dataset=Cora --loss=info --encoder=GCN --tau=0.6 --act=prelu --num_hidden=128 --num_proj_hidden=128 --aug_type=noisy_z --std=1e-5 --lr=0.001 --num_epochs=1000
python train.py --gpu_id=$DEVICE --dataset=CiteSeer --loss=info --encoder=GCN --tau=0.5 --act=prelu --num_hidden=256 --num_proj_hidden=256 --num_epochs=0
python train.py --gpu_id=$DEVICE --dataset=CiteSeer --loss=info --encoder=GCN --tau=0.5 --act=prelu --num_hidden=256 --num_proj_hidden=256 --pe1=0.3 --pe2=0.2 --pf1=0.3 --pf2=0.3 --lr=0.001 --num_epochs=200
python train.py --gpu_id=$DEVICE --dataset=CiteSeer --loss=uniform --encoder=GCN --tau=0.9 --act=prelu --num_hidden=256 --num_proj_hidden=256 --pe1=0.0 --pe2=0.0 --pf1=0.0 --pf2=0.4 --lr=0.001 --num_epochs=200
python train.py --gpu_id=$DEVICE --dataset=CiteSeer --loss=align --encoder=GCN --tau=0.5 --act=prelu --num_hidden=256 --num_proj_hidden=256 --pe1=0.3 --pe2=0.2 --pf1=0.3 --pf2=0.3 --lr=0.001 --num_epochs=200
python train.py --gpu_id=$DEVICE --dataset=CiteSeer --loss=align --encoder=GCN --tau=0.8 --act=prelu --num_hidden=256 --num_proj_hidden=256 --pe1=0.0 --pe2=0.0 --pf1=0.4 --pf2=0.0 --lr=0.001 --num_epochs=200 --use_norm --norm_type=cn --scale=5
python train.py --gpu_id=$DEVICE --dataset=CiteSeer --loss=info --encoder=GCN --tau=0.5 --act=prelu --num_hidden=256 --num_proj_hidden=256 --aug_type=none --lr=0.001 --num_epochs=200
python train.py --gpu_id=$DEVICE --dataset=CiteSeer --loss=info --encoder=GCN --tau=0.9 --act=prelu --num_hidden=256 --num_proj_hidden=256 --aug_type=noisy_z --std=1e-4 --lr=0.001 --num_epochs=200
python train.py --gpu_id=$DEVICE --dataset=PubMed --loss=info --encoder=GCN --tau=0.2 --act=prelu --num_hidden=256 --num_proj_hidden=256 --num_epochs=0
python train.py --gpu_id=$DEVICE --dataset=PubMed --loss=info --encoder=GCN --tau=0.2 --act=prelu --num_hidden=256 --num_proj_hidden=256 --pe1=0.0 --pe2=0.2 --pf1=0.2 --pf2=0.1 --lr=0.001 --num_epochs=1500
python train.py --gpu_id=$DEVICE --dataset=PubMed --loss=uniform --encoder=GCN --tau=0.2 --act=prelu --num_hidden=256 --num_proj_hidden=256 --pe1=0.0 --pe2=0.2 --pf1=0.2 --pf2=0.1 --lr=0.001 --num_epochs=1500
python train.py --gpu_id=$DEVICE --dataset=PubMed --loss=align --encoder=GCN --tau=0.2 --act=prelu --num_hidden=256 --num_proj_hidden=256 --pe1=0.0 --pe2=0.2 --pf1=0.2 --pf2=0.1 --lr=0.001 --num_epochs=1500
python train.py --gpu_id=$DEVICE --dataset=PubMed --loss=align --encoder=GCN --tau=0.2 --act=prelu --num_hidden=256 --num_proj_hidden=256 --pe1=0.3 --pe2=0.3 --pf1=0.2 --pf2=0.2 --lr=0.001 --num_epochs=1000 --norm_type=dcn --scale=1000 --use_norm
python train.py --gpu_id=$DEVICE --dataset=PubMed --loss=info --encoder=GCN --tau=0.9 --act=prelu --num_hidden=256 --num_proj_hidden=256 --aug_type=none --lr=0.001 --num_epochs=1500
python train.py --gpu_id=$DEVICE --dataset=PubMed --loss=info --encoder=GCN --tau=0.3 --act=prelu --num_hidden=256 --num_proj_hidden=256 --aug_type=noisy_z --std=1e-5 --lr=0.001 --num_epochs=1500
python train.py --gpu_id=$DEVICE --dataset=Photo --loss=info --encoder=GCN --tau=0.6 --act=prelu --num_hidden=128 --num_proj_hidden=128 --num_epochs=0
python train.py --gpu_id=$DEVICE --dataset=Photo --loss=info --encoder=GCN --tau=0.6 --act=prelu --num_hidden=128 --num_proj_hidden=128 --pe1=0.1 --pe2=0.4 --pf1=0.0 --pf2=0.2 --lr=0.01 --num_epochs=1000
python train.py --gpu_id=$DEVICE --dataset=Photo --loss=uniform --encoder=GCN --tau=0.1 --act=prelu --num_hidden=128 --num_proj_hidden=128 --pe1=0.1 --pe2=0.1 --pf1=0.0 --pf2=0.1 --lr=0.001 --num_epochs=1000
python train.py --gpu_id=$DEVICE --dataset=Photo --loss=align --encoder=GCN --tau=0.6 --act=prelu --num_hidden=128 --num_proj_hidden=128 --pe1=0.1 --pe2=0.4 --pf1=0.0 --pf2=0.2 --lr=0.01 --num_epochs=1000
python train.py --gpu_id=$DEVICE --dataset=Photo --loss=align --encoder=GCN --tau=0.9 --act=prelu --num_hidden=128 --num_proj_hidden=128 --pe1=0.2 --pe2=0.2 --pf1=0.2 --pf2=0.2 --lr=0.001 --num_epochs=1500 --norm_type=cn --scale=1000 --use_norm
python train.py --gpu_id=$DEVICE --dataset=Photo --loss=info --encoder=GCN --tau=0.1 --act=prelu --num_hidden=128 --num_proj_hidden=128 --aug_type=none --lr=0.001 --num_epochs=1000
python train.py --gpu_id=$DEVICE --dataset=Photo --loss=info --encoder=GCN --tau=0.1 --act=prelu --num_hidden=128 --num_proj_hidden=128 --aug_type=noisy_z --std=1e-5 --lr=0.001 --num_epochs=1000
python train.py --gpu_id=$DEVICE --dataset=Computers --loss=info --encoder=GCN --tau=0.1 --act=prelu --num_hidden=128 --num_proj_hidden=128 --num_epochs=0
python train.py --gpu_id=$DEVICE --dataset=Computers --loss=info --encoder=GCN --tau=0.1 --act=prelu --num_hidden=128 --num_proj_hidden=128 --pe1=0.2 --pe2=0.2 --pf1=0.2 --pf2=0.2 --lr=5e-4 --num_epochs=5000
python train.py --gpu_id=$DEVICE --dataset=Computers --loss=uniform --encoder=GCN --tau=0.1 --act=prelu --num_hidden=128 --num_proj_hidden=128 --pe1=0.0 --pe2=0.2 --pf1=0.4 --pf2=0.0 --lr=5e-4 --num_epochs=5000
python train.py --gpu_id=$DEVICE --dataset=Computers --loss=align --encoder=GCN --tau=0.1 --act=prelu --num_hidden=128 --num_proj_hidden=128 --pe1=0.2 --pe2=0.2 --pf1=0.2 --pf2=0.2 --lr=5e-4 --num_epochs=5000
python train.py --gpu_id=$DEVICE --dataset=Computers --loss=align --encoder=GCN --tau=1.0 --act=prelu --num_hidden=128 --num_proj_hidden=128 --pe1=0.1 --pe2=0.3 --pf1=0.0 --pf2=0.4 --lr=1e-5 --num_epochs=2000 --norm_type=dcn --scale=50 --use_norm
python train.py --gpu_id=$DEVICE --dataset=Computers --loss=info --encoder=GCN --tau=0.7 --act=prelu --num_hidden=128 --num_proj_hidden=128 --aug_type=none --lr=5e-4 --num_epochs=5000
python train.py --gpu_id=$DEVICE --dataset=Computers --loss=info --encoder=GCN --tau=0.4 --act=prelu --num_hidden=128 --num_proj_hidden=128 --aug_type=noisy_z --std=1e-5 --lr=0.001 --num_epochs=5000