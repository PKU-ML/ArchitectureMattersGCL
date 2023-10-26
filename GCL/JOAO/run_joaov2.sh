DEVICE=$1
python joaov2.py --gpu_id=$DEVICE --dataset=MUTAG --num_epochs=0 --hidden_dim=32 --num_layers=5
python joaov2.py --gpu_id=$DEVICE --dataset=MUTAG --loss=info --lr=0.01 --hidden_dim=32 --num_epochs=250 --num_layers=5 --tau=1.0 --gamma=1
python joaov2.py --gpu_id=$DEVICE --dataset=MUTAG --loss=uniform --lr=0.001 --hidden_dim=32 --num_epochs=200 --num_layers=5 --tau=1.0 --gamma=1
python joaov2.py --gpu_id=$DEVICE --dataset=MUTAG --loss=align --lr=0.01 --hidden_dim=32 --num_epochs=250 --num_layers=5 --tau=1.0 --gamma=0.01

python joaov2.py --gpu_id=$DEVICE --dataset=PTC_MR --num_epochs=0 --hidden_dim=32 --num_layers=5
python joaov2.py --gpu_id=$DEVICE --dataset=PTC_MR --loss=info --lr=0.001 --hidden_dim=32 --num_epochs=200 --num_layers=5 --tau=1.0 --gamma=1
python joaov2.py --gpu_id=$DEVICE --dataset=PTC_MR --loss=uniform --lr=0.001 --hidden_dim=32 --num_epochs=20 --num_layers=5 --tau=1.0 --gamma=1
python joaov2.py --gpu_id=$DEVICE --dataset=PTC_MR --loss=align --lr=0.01 --hidden_dim=32 --num_epochs=100 --num_layers=5 --tau=1.0 --gamma=1

python joaov2.py --gpu_id=$DEVICE --dataset=IMDB-BINARY --num_epochs=0 --hidden_dim=32 --num_layers=5
python joaov2.py --gpu_id=$DEVICE --dataset=IMDB-BINARY --loss=info --lr=0.001 --hidden_dim=32 --num_epochs=10 --num_layers=3 --tau=1.0 --gamma=0.01
python joaov2.py --gpu_id=$DEVICE --dataset=IMDB-BINARY --loss=uniform --lr=0.001 --hidden_dim=32 --num_epochs=10 --num_layers=3 --tau=1.0 --gamma=0.01
python joaov2.py --gpu_id=$DEVICE --dataset=IMDB-BINARY --loss=align --lr=0.001 --hidden_dim=32 --num_epochs=10 --num_layers=3 --tau=1.0 --gamma=0.01

python joaov2.py --gpu_id=$DEVICE --dataset=IMDB-MULTI --num_epochs=0 --hidden_dim=32 --num_layers=3
python joaov2.py --gpu_id=$DEVICE --dataset=IMDB-MULTI --loss=info --lr=0.01 --hidden_dim=32 --num_epochs=10 --num_layers=3 --tau=1.0 --gamma=0.01
python joaov2.py --gpu_id=$DEVICE --dataset=IMDB-MULTI --loss=uniform --lr=0.01 --hidden_dim=32 --num_epochs=10 --num_layers=3 --tau=1.0 --gamma=0.01
python joaov2.py --gpu_id=$DEVICE --dataset=IMDB-MULTI --loss=align --lr=0.01 --hidden_dim=32 --num_epochs=10 --num_layers=3 --tau=1.0 --gamma=0.01

python joaov2.py --gpu_id=$DEVICE --dataset=PROTEINS --num_epochs=0 --hidden_dim=32 --num_layers=5
python joaov2.py --gpu_id=$DEVICE --dataset=PROTEINS --loss=info --lr=0.01 --hidden_dim=32 --num_epochs=200 --num_layers=5 --tau=1.0 --gamma=1
python joaov2.py --gpu_id=$DEVICE --dataset=PROTEINS --loss=uniform --lr=0.01 --hidden_dim=32 --num_epochs=200 --num_layers=5 --tau=1.0 --gamma=1
python joaov2.py --gpu_id=$DEVICE --dataset=PROTEINS --loss=align --lr=0.001 --hidden_dim=32 --num_epochs=200 --num_layers=5 --tau=1.0 --gamma=1

python joaov2.py --gpu_id=$DEVICE --dataset=REDDIT-BINARY --num_epochs=0 --hidden_dim=32 --num_layers=5
python joaov2.py --gpu_id=$DEVICE --dataset=REDDIT-BINARY --loss=info --lr=0.001 --hidden_dim=32 --num_epochs=20 --num_layers=5 --tau=1.0 --gamma=0.1
python joaov2.py --gpu_id=$DEVICE --dataset=REDDIT-BINARY --loss=uniform --lr=0.001 --hidden_dim=32 --num_epochs=20 --num_layers=5 --tau=1.0 --gamma=0.1
python joaov2.py --gpu_id=$DEVICE --dataset=REDDIT-BINARY --loss=align --lr=0.001 --hidden_dim=32 --num_epochs=20 --num_layers=5 --tau=1.0 --gamma=1																				