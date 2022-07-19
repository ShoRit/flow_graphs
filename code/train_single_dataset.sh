#! /bin/bash

set -euxo pipefail

RAYON_RS_NUM_CPUS=5

for (( SEED = 0 ; SEED <= 2; SEED++ ))
do
    for (( DEP=0 ; DEP <= 1; DEP++ ))
    do
        for (( AMR=0 ; AMR <= 1; AMR++ ))
        do
            python rel_classification_info.py --src_dataset $1 --tgt_dataset $1 --mode train --domain src --dep $DEP --amr $AMR --gnn rgcn --seed $SEED --batch_size 16 --gnn_depth 4 --gpu 0 --lr 2e-5
        done
    done
done