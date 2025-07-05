#!/bin/bash
for seed in {0..40..10}
do
    echo "Running seed ${seed}"
    python run_inference.py \
        --dataset datasets/test_features_mutual-information_seed${seed}.csv \
        --model models/mi_seed${seed}.pkl \
        --save_filename test_inference_time_mi_seed${seed}.csv \
        --num_exec 100
done