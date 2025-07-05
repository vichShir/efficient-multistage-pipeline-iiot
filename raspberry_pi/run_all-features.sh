#!/bin/bash
for seed in {0..40..10}
do
    echo "Running seed ${seed}"
    python run_inference.py \
        --dataset datasets/test_263-features_seed${seed}.csv \
        --model models/training_size_100.0_seed${seed}.pkl \
        --save_filename test_inference_time_all-features_seed${seed}.csv \
        --num_exec 100
done