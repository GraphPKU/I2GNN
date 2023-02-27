#!/bin/bash
h_size=(1 2 2 3)
seed=1
# i: 0 (3-cycle), 1 (4-cycle), 2 (5-cycle), 3 (6-cycle)
for i in 0 1 2 3
do
	python -u run_count.py --h ${h_size[i]} --target $i --dataset count_cycle --model $1 --batch_size 256 --epoch 2000 --seed ${seed}
done


h_size=(2 2 1 4 2)
# i: 0 (tailed triangle), 1 (chordal triangle), 2 (4-clique), 3 (4-path), 4 (triangle-rectangle)
for i in 0 1 2 3 4
do
  python -u run_count.py --h ${h_size[i]} --target $i --dataset count_graphlet --model $1 --batch_size 256 --epoch 2000 --seed ${seed}
done
