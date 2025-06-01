#!/bin/bash
for seed in {0..40..10}
do
    echo "Saving feature files for seed ${seed}"
    python ../generate_feature_files.py --seed ${seed} --split test
done