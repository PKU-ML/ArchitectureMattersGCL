DEVICE=$1
python main.py --gpu_id=$DEVICE --DS=MUTAG --hidden_dim=512 --num_epochs=0 --num_layers=4
python main.py --gpu_id=$DEVICE --DS=MUTAG --loss=jsd --lr=0.01 --hidden_dim=512 --num_epochs=100 --num_layers=4
python main.py --gpu_id=$DEVICE --DS=MUTAG --loss=neg --lr=0.01 --hidden_dim=32 --num_epochs=10 --num_layers=12
python main.py --gpu_id=$DEVICE --DS=MUTAG --loss=pos --lr=0.0001 --hidden_dim=32 --num_epochs=10 --num_layers=8																						

python main.py --gpu_id=$DEVICE --DS=PTC_MR --hidden_dim=512 --num_epochs=0 --num_layers=4
python main.py --gpu_id=$DEVICE --DS=PTC_MR --loss=jsd --lr=0.001 --hidden_dim=32 --num_epochs=20 --num_layers=8
python main.py --gpu_id=$DEVICE --DS=PTC_MR --loss=neg --lr=0.01 --hidden_dim=32 --num_epochs=20 --num_layers=8
python main.py --gpu_id=$DEVICE --DS=PTC_MR --loss=pos --lr=0.01 --hidden_dim=512 --num_epochs=20 --num_layers=8

python main.py --gpu_id=$DEVICE --DS=IMDB-BINARY --hidden_dim=32 --num_epochs=0 --num_layers=8
python main.py --gpu_id=$DEVICE --DS=IMDB-BINARY --loss=jsd --lr=0.001 --hidden_dim=32 --num_epochs=10 --num_layers=8
python main.py --gpu_id=$DEVICE --DS=IMDB-BINARY --loss=neg --lr=0.001 --hidden_dim=32 --num_epochs=20 --num_layers=8
python main.py --gpu_id=$DEVICE --DS=IMDB-BINARY --loss=pos --lr=0.01 --hidden_dim=512 --num_epochs=10 --num_layers=12

python main.py --gpu_id=$DEVICE --DS=IMDB-MULTI --hidden_dim=32 --num_epochs=0 --num_layers=8
python main.py --gpu_id=$DEVICE --DS=IMDB-MULTI --loss=jsd --lr=0.0001 --hidden_dim=32 --num_epochs=10 --num_layers=8
python main.py --gpu_id=$DEVICE --DS=IMDB-MULTI --loss=neg --lr=0.01 --hidden_dim=32 --num_epochs=100 --num_layers=12
python main.py --gpu_id=$DEVICE --DS=IMDB-MULTI --loss=pos --lr=0.01 --hidden_dim=512 --num_epochs=20 --num_layers=4

python main.py --gpu_id=$DEVICE --DS=PROTEINS --hidden_dim=32 --num_epochs=0 --num_layers=5
python main.py --gpu_id=$DEVICE --DS=PROTEINS --loss=jsd --lr=0.01 --hidden_dim=32 --num_epochs=20 --num_layers=5
python main.py --gpu_id=$DEVICE --DS=PROTEINS --loss=neg --lr=0.01 --hidden_dim=32 --num_epochs=20 --num_layers=5
python main.py --gpu_id=$DEVICE --DS=PROTEINS --loss=pos --lr=0.01 --hidden_dim=32 --num_epochs=20 --num_layers=5

python main.py --gpu_id=$DEVICE --DS=REDDIT-BINARY --hidden_dim=32 --num_epochs=0 --num_layers=5
python main.py --gpu_id=$DEVICE --DS=REDDIT-BINARY --loss=jsd --lr=0.001 --hidden_dim=32 --num_epochs=20 --num_layers=5
python main.py --gpu_id=$DEVICE --DS=REDDIT-BINARY --loss=neg --lr=0.001 --hidden_dim=32 --num_epochs=20 --num_layers=5
python main.py --gpu_id=$DEVICE --DS=REDDIT-BINARY --loss=pos --lr=0.001 --hidden_dim=32 --num_epochs=20 --num_layers=5