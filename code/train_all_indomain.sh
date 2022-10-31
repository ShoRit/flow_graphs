#!/bin/bash

set -euxo pipefail

DATASET=risec
EXPERIMENT_CONFIG=amr_residual

for SEED in 0 1 2
do
    sbatch --partition=shire_general --gres=gpu:A4500:1 ./singularity_runner.sh ./train_single_indomain_model.sh --dataset $DATASET --seed $SEED --experiment_config $EXPERIMENT_CONFIG 
done
