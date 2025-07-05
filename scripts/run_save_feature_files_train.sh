#!/bin/bash
for seed in {0..40..10}
do
    echo "Saving feature files for seed ${seed}"
    read -p "Please enter the training size proportion for seed ${seed}: " prop
    python generate_feature_files.py --seed ${seed} --split train --filename BRUIIoT_train_size${prop}_seed${seed}.csv
done