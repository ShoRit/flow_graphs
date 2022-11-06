#!/bin/bash 

set -euxo pipefail # fail for each command
source ~/.bashrc; 
conda activate flow_graphs; 
python3 eval_transfer_model.py --src_dataset mscorpus   --tgt_dataset risec     --fewshot 0.01 --seed 0 --experiment_config amr_residual