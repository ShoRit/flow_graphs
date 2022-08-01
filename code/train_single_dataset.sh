#! /bin/bash

set -euxo pipefail

RAYON_RS_NUM_CPUS=5

for (( SEED = 0 ; SEED <= 2; SEED++ ))
do
    for VALUE in dep amr
    do
            echo python train_single_domain_model.py --dataset risec --seed $SEED --experiment_config "$VALUE"_residual --gpu None
    done
done