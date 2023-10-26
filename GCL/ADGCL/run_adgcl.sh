DEVICE=$1
python test_minmax_tu.py --gpu_id=$DEVICE --dataset=MUTAG --batch_size=32 --epochs=0
python test_minmax_tu.py --gpu_id=$DEVICE --dataset=MUTAG --loss=info --batch_size=32 --epochs=100 --model_lr=0.001 --reg_lambda=10.0 
python test_minmax_tu.py --gpu_id=$DEVICE --dataset=MUTAG --loss=uniform --batch_size=128 --epochs=200 --model_lr=0.001 --reg_lambda=5.0
python test_minmax_tu.py --gpu_id=$DEVICE --dataset=MUTAG --loss=align --batch_size=128 --epochs=100 --model_lr=0.001 --reg_lambda=2.0

python test_minmax_tu.py --gpu_id=$DEVICE --dataset=PTC_MR --batch_size=32 --epochs=0
python test_minmax_tu.py --gpu_id=$DEVICE --dataset=PTC_MR --loss=info --batch_size=32 --epochs=20 --model_lr=0.001 --reg_lambda=2.0 
python test_minmax_tu.py --gpu_id=$DEVICE --dataset=PTC_MR --loss=uniform --batch_size=32 --epochs=20 --model_lr=0.001 --reg_lambda=2.0
python test_minmax_tu.py --gpu_id=$DEVICE --dataset=PTC_MR --loss=align --batch_size=128 --epochs=50 --model_lr=0.001 --reg_lambda=5.0

python test_minmax_tu.py --gpu_id=$DEVICE --dataset=IMDB-BINARY --batch_size=32 --epochs=0
python test_minmax_tu.py --gpu_id=$DEVICE --dataset=IMDB-BINARY --loss=info --batch_size=32 --epochs=100 --model_lr=0.001 --reg_lambda=2.0 
python test_minmax_tu.py --gpu_id=$DEVICE --dataset=IMDB-BINARY --loss=uniform --batch_size=32 --epochs=100 --model_lr=0.001 --reg_lambda=1.0
python test_minmax_tu.py --gpu_id=$DEVICE --dataset=IMDB-BINARY --loss=align --batch_size=32 --epochs=100 --model_lr=0.001 --reg_lambda=1.0

python test_minmax_tu.py --gpu_id=$DEVICE --dataset=IMDB-MULTI --batch_size=32 --epochs=0
python test_minmax_tu.py --gpu_id=$DEVICE --dataset=IMDB-MULTI --loss=info --batch_size=32 --epochs=100 --model_lr=0.001 --reg_lambda=10.0 
python test_minmax_tu.py --gpu_id=$DEVICE --dataset=IMDB-MULTI --loss=uniform --batch_size=128 --epochs=200 --model_lr=0.001 --reg_lambda=2.0
python test_minmax_tu.py --gpu_id=$DEVICE --dataset=IMDB-MULTI --loss=align --batch_size=32 --epochs=150 --model_lr=0.001 --reg_lambda=10.0

python test_minmax_tu.py --gpu_id=$DEVICE --dataset=PROTEINS --batch_size=32 --epochs=0
python test_minmax_tu.py --gpu_id=$DEVICE --dataset=PROTEINS --loss=info --batch_size=32 --epochs=200 --model_lr=0.01 --reg_lambda=2.0 
python test_minmax_tu.py --gpu_id=$DEVICE --dataset=PROTEINS --loss=uniform --batch_size=32 --epochs=100 --model_lr=0.01 --reg_lambda=10.0
python test_minmax_tu.py --gpu_id=$DEVICE --dataset=PROTEINS --loss=align --batch_size=32 --epochs=20 --model_lr=0.01 --reg_lambda=2.0

python test_minmax_tu.py --gpu_id=$DEVICE --dataset=REDDIT-BINARY --batch_size=32 --epochs=0
python test_minmax_tu.py --gpu_id=$DEVICE --dataset=REDDIT-BINARY --loss=info --batch_size=32 --epochs=100 --model_lr=0.001 --reg_lambda=10.0 
python test_minmax_tu.py --gpu_id=$DEVICE --dataset=REDDIT-BINARY --loss=uniform --batch_size=32 --epochs=200 --model_lr=0.001 --reg_lambda=5.0
python test_minmax_tu.py --gpu_id=$DEVICE --dataset=REDDIT-BINARY --loss=align --batch_size=32 --epochs=100 --model_lr=0.01 --reg_lambda=2.0