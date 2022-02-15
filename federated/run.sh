#!/usr/bin/env bash
python3  -u main.py --dataset='stackoverflow' \
        --optimizer=fedavg \
        --scale=0 \
        --clipping_bound=2 \
	      --learning_rate=1 \
	      --sigma=0.3 \
	      --delta=0.0025 \
        --num_rounds=40 \
        --clients_per_round=20 \
        --eval_every=1 \
        --batch_size=100 \
        --num_epochs=1 \
        --model='lr' \


