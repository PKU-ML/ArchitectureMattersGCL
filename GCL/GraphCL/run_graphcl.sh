DEVICE=$1

python gsimclr.py --gpu_id=$DEVICE --dataset=MUTAG --num_epochs=0 --hidden_dim=512
python gsimclr.py --gpu_id=$DEVICE --dataset=MUTAG --loss=info --num_epochs=100 --lr=0.001 --hidden_dim=512 --num_layers=8 --tau=1.0
python gsimclr.py --gpu_id=$DEVICE --dataset=MUTAG --loss=uniform --num_epochs=100 --lr=0.001 --hidden_dim=512 --num_layers=4 --tau=1.0
python gsimclr.py --gpu_id=$DEVICE --dataset=MUTAG --loss=align --num_epochs=10 --lr=0.001 --hidden_dim=512 --num_layers=8 --tau=0.1	

python gsimclr.py --gpu_id=$DEVICE --dataset=PTC_MR --num_epochs=0 --hidden_dim=32
python gsimclr.py --gpu_id=$DEVICE --dataset=PTC_MR --loss=info --num_epochs=10 --lr=0.001 --hidden_dim=32 --num_layers=4 --tau=0.2
python gsimclr.py --gpu_id=$DEVICE --dataset=PTC_MR --loss=uniform --num_epochs=10 --lr=0.001 --hidden_dim=32 --num_layers=4 --tau=0.5
python gsimclr.py --gpu_id=$DEVICE --dataset=PTC_MR --loss=align --num_epochs=20 --lr=0.0001 --hidden_dim=32 --num_layers=4 --tau=0.4
				
python gsimclr.py --gpu_id=$DEVICE --dataset=PTC_MR --num_epochs=0 --hidden_dim=32
python gsimclr.py --gpu_id=$DEVICE --dataset=PTC_MR --loss=info --num_epochs=10 --lr=0.001 --hidden_dim=32 --num_layers=4 --tau=0.2
python gsimclr.py --gpu_id=$DEVICE --dataset=PTC_MR --loss=uniform --num_epochs=10 --lr=0.001 --hidden_dim=32 --num_layers=4 --tau=0.5
python gsimclr.py --gpu_id=$DEVICE --dataset=PTC_MR --loss=align --num_epochs=20 --lr=0.0001 --hidden_dim=32 --num_layers=4 --tau=0.4

python gsimclr.py --gpu_id=$DEVICE --dataset=IMDB-BINARY --num_epochs=0 --hidden_dim=32
python gsimclr.py --gpu_id=$DEVICE --dataset=IMDB-BINARY --loss=info --num_epochs=10 --lr=0.001 --hidden_dim=32 --num_layers=4 --tau=0.2
python gsimclr.py --gpu_id=$DEVICE --dataset=IMDB-BINARY --loss=uniform --num_epochs=100 --lr=0.001 --hidden_dim=32 --num_layers=4 --tau=0.9
python gsimclr.py --gpu_id=$DEVICE --dataset=IMDB-BINARY --loss=align --num_epochs=20 --lr=0.001 --hidden_dim=32 --num_layers=8 --tau=0.2

python gsimclr.py --gpu_id=$DEVICE --dataset=IMDB-MULTI --num_epochs=0 --hidden_dim=32
python gsimclr.py --gpu_id=$DEVICE --dataset=IMDB-MULTI --loss=info --num_epochs=20 --lr=0.001 --hidden_dim=32 --num_layers=4 --tau=1.0
python gsimclr.py --gpu_id=$DEVICE --dataset=IMDB-MULTI --loss=uniform --num_epochs=100 --lr=0.001 --hidden_dim=32 --num_layers=8 --tau=0.8
python gsimclr.py --gpu_id=$DEVICE --dataset=IMDB-MULTI --loss=align --num_epochs=20 --lr=0.001 --hidden_dim=32 --num_layers=8 --tau=0.3

python gsimclr.py --gpu_id=$DEVICE --dataset=PROTEINS --num_epochs=0 --hidden_dim=32
python gsimclr.py --gpu_id=$DEVICE --dataset=PROTEINS --loss=info --num_epochs=100 --lr=0.001 --hidden_dim=32 --num_layers=4 --tau=0.4
python gsimclr.py --gpu_id=$DEVICE --dataset=PROTEINS --loss=uniform --num_epochs=100 --lr=0.01 --hidden_dim=32 --num_layers=4 --tau=1.0
python gsimclr.py --gpu_id=$DEVICE --dataset=PROTEINS --loss=align --num_epochs=100 --lr=0.01 --hidden_dim=32 --num_layers=4 --tau=0.8

python gsimclr.py --gpu_id=$DEVICE --dataset=REDDIT-BINARY --num_epochs=0 --hidden_dim=32
python gsimclr.py --gpu_id=$DEVICE --dataset=REDDIT-BINARY --loss=info --num_epochs=100 --lr=0.001 --hidden_dim=32 --num_layers=4 --tau=0.8
python gsimclr.py --gpu_id=$DEVICE --dataset=REDDIT-BINARY --loss=uniform --num_epochs=10 --lr=0.01 --hidden_dim=32 --num_layers=4 --tau=1.0
python gsimclr.py --gpu_id=$DEVICE --dataset=REDDIT-BINARY --loss=align --num_epochs=100 --lr=0.001 --hidden_dim=32 --num_layers=4 --tau=0.8