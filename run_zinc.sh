seed=0
python -u run_zinc.py --model I2GNN --dataset $1 --h 3 --node_label spd --use_rd --batch_size 256 --epoch 1000 --seed ${seed}
