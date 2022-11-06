#!/bin/bash

set -euxo pipefail

for SRC_DATASET in japflow mscorpus
do
    for TGT_DATASET in risec
    do
        for FEWSHOT in 1 10 50 100
        do
            for SEED in 0 1 2
            do
                for EXPERIMENT_CONFIG in amr_residual dep_residual baseline
                do
                    sbatch --partition=shire_general --gres=gpu:1080Ti:1 ./singularity_runner.sh ./eval_single_transfer_model.sh $SRC_DATASET $TGT_DATASET $FEWSHOT $SEED $EXPERIMENT_CONFIG
                done
            done
        done
    done
done