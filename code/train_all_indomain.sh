#!/bin/bash

set -euxo pipefail

DATASET=risec
EXPERIMENT_CONFIG=amr_residual

for SEED in 0 1 2
do
    srun --pty --partition=shire_general --gres=gpu:A4500:1 ./singularity_runner.sh ./train_single_indomain_model.sh  $DATASET $SEED $EXPERIMENT_CONFIG 
done
