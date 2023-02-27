#!/bin/bash

# for seed in 0 1 2 3 4 5 6 7 8 9
# do
seed=0
python -u run_ogb_mol.py --h 4 --use_pooling_nn --num_layer 6 --node_label spd --use_rd --drop_ratio 0.35 --epochs 50 --seed ${seed} --scheduler
# done
