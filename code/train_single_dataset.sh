#! /bin/bash

set -euxo pipefail

for (( SEED = 0 ; SEED <= 2; SEED++ ))
do
    python rel_classification_info.py --src_dataset $1 --tgt_dataset $1 --mode train --domain src --dep 1 --amr 1 --gnn rgcn --seed $SEED --alpha 0.5 --batch_size 16 --gnn_depth 2 --gpu 0
done