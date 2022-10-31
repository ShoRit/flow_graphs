#!/bin/bash

set -euxo pipefail

for DATASET in risec japflow mscorpus
do
    for SEED in 0 1 2
    do
        for CASE in amr_residual dep_residual plaintext
        do
            CHECKPOINT_FILE=../checkpoints/indomain-$DATASET-$CASE-rgcn-depth_4-seed_$SEED-lr_2e-05.pt
            if [[ ! -f $CHECKPOINT_FILE ]]
            then
                dvc pull $CHECKPOINT_FILE
            fi
            sbatch --partition=shire_general --gres=gpu:1080Ti:1 ./singularity_runner.sh ./eval_single_indomain_model.sh $CHECKPOINT_FILE
        done
    done
done