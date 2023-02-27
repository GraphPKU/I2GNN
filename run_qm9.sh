#!/bin/bash
h=3
for t in 1 2 3 5 6 9 10 11
do
	python -u run_qm9.py --h $h --dataset qm9 --model I2GNN --target $t --batch_size 64 --node_label spd --use_rd --use_pooling_nn --epoch 350
done


for t in 4 7 8
do
	python -u run_qm9.py --h $h --dataset qm9 --model I2GNN --target $t --batch_size 64 --node_label spd --use_rd --epoch 350
done

for t in 0
do
	python -u run_qm9.py --h $h --dataset qm9 --model I2GNN --target $t --batch_size 64 --node_label spd --use_rd --use_pooling_nn --lr_decay_factor 0.7 --epoch 150
done 
