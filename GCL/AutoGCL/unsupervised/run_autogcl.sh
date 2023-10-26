DEVICE=$1
python us_main.py --gpu=$DEVICE --dataset=PTC_MR --loss=info --lr=0.001 --hidden-dim=128 --tau=1.0 --epochs=0
python us_main.py --gpu=$DEVICE --dataset=PTC_MR --loss=info --lr=0.001 --hidden-dim=128 --tau=0.1 --epochs=30
python us_main.py --gpu=$DEVICE --dataset=PTC_MR --loss=uniform --lr=0.001 --hidden-dim=128 --tau=1.0 --epochs=50
python us_main.py --gpu=$DEVICE --dataset=PTC_MR --loss=align --lr=0.001 --hidden-dim=128 --tau=1.0 --epochs=30

python us_main.py --gpu=$DEVICE --dataset=IMDB-BINARY --loss=info --lr=0.001 --hidden-dim=128 --tau=1.0 --epochs=0
python us_main.py --gpu=$DEVICE --dataset=IMDB-BINARY --loss=info --lr=0.001 --hidden-dim=128 --tau=1.0 --epochs=30
python us_main.py --gpu=$DEVICE --dataset=IMDB-BINARY --loss=uniform --lr=0.001 --hidden-dim=128 --tau=1.0 --epochs=30
python us_main.py --gpu=$DEVICE --dataset=IMDB-BINARY --loss=align --lr=0.001 --hidden-dim=128 --tau=1.0 --epochs=30

python us_main.py --gpu=$DEVICE --dataset=IMDB-MULTI --loss=info --lr=0.001 --hidden-dim=128 --tau=1.0 --epochs=0
python us_main.py --gpu=$DEVICE --dataset=IMDB-MULTI --loss=info --lr=0.001 --hidden-dim=128 --tau=1.0 --epochs=30
python us_main.py --gpu=$DEVICE --dataset=IMDB-MULTI --loss=uniform --lr=0.001 --hidden-dim=128 --tau=1.0 --epochs=30
python us_main.py --gpu=$DEVICE --dataset=IMDB-MULTI --loss=align --lr=0.001 --hidden-dim=128 --tau=1.0 --epochs=30

python us_main.py --gpu=$DEVICE --dataset=REDDIT-BINARY --loss=info --lr=0.001 --hidden-dim=128 --tau=1.0 --epochs=0
python us_main.py --gpu=$DEVICE --dataset=REDDIT-BINARY --loss=info --lr=0.001 --hidden-dim=128 --tau=1.0 --epochs=30
python us_main.py --gpu=$DEVICE --dataset=REDDIT-BINARY --loss=uniform --lr=0.001 --hidden-dim=128 --tau=1.0 --epochs=30
python us_main.py --gpu=$DEVICE --dataset=REDDIT-BINARY --loss=align --lr=0.001 --hidden-dim=128 --tau=1.0 --epochs=30

python us_main.py --gpu=$DEVICE --dataset=PROTEINS --loss=info --lr=0.001 --hidden-dim=128 --tau=1.0 --epochs=0
python us_main.py --gpu=$DEVICE --dataset=PROTEINS --loss=info --lr=0.001 --hidden-dim=128 --tau=1.0 --epochs=30
python us_main.py --gpu=$DEVICE --dataset=PROTEINS --loss=uniform --lr=0.001 --hidden-dim=128 --tau=1.0 --epochs=30
python us_main.py --gpu=$DEVICE --dataset=PROTEINS --loss=align --lr=0.001 --hidden-dim=128 --tau=1.0 --epochs=30

python us_main.py --gpu=$DEVICE --dataset=MUTAG --loss=info --lr=0.001 --hidden-dim=128 --tau=1.0 --epochs=0 
python us_main.py --gpu=$DEVICE --dataset=MUTAG --loss=info --lr=0.001 --hidden-dim=128 --tau=0.1 --epochs=30
python us_main.py --gpu=$DEVICE --dataset=MUTAG --loss=uniform --lr=0.001 --hidden-dim=128 --tau=0.1 --epochs=30
python us_main.py --gpu=$DEVICE --dataset=MUTAG --loss=align --lr=0.001 --hidden-dim=128 --tau=0.1 --epochs=30