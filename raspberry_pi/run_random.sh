#!/bin/bash

NUM_FEATS=19

for seed in {0..40..10}
do
    echo "Running seed ${seed}"
    python run_inference.py \
        --dataset datasets/test_features_random_seed${seed}.csv \
        --model models/random-${NUM_FEATS}_seed${seed}.pkl \
        --save_filename test_inference_time_random_seed${seed}.csv \
        --num_exec 100
done