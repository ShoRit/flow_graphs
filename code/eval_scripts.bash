#!/bin/bash 

# set -euxo pipefail # fail for each command

for fewshot in 0.01 0.05 0.1 0.2 0.5
do
    for seed in 0 1 2
    do  
        python3 eval_transfer_model.py --src_dataset mscorpus   --tgt_dataset risec     --fewshot $fewshot --seed $seed --experiment_config baseline
        python3 eval_transfer_model.py --src_dataset risec      --tgt_dataset mscorpus  --fewshot $fewshot --seed $seed --experiment_config baseline
        python3 eval_transfer_model.py --src_dataset japflow    --tgt_dataset risec     --fewshot $fewshot --seed $seed --experiment_config baseline
        python3 eval_transfer_model.py --src_dataset risec      --tgt_dataset japflow   --fewshot $fewshot --seed $seed --experiment_config baseline
        python3 eval_transfer_model.py --src_dataset japflow    --tgt_dataset mscorpus  --fewshot $fewshot --seed $seed --experiment_config baseline
        python3 eval_transfer_model.py --src_dataset mscorpus   --tgt_dataset japflow   --fewshot $fewshot --seed $seed --experiment_config baseline
        # python3 eval_transfer_model.py --src_dataset mscorpus   --tgt_dataset risec     --fewshot $fewshot --seed $seed --experiment_config baseline
        # python3 eval_transfer_model.py --src_dataset risec      --tgt_dataset mscorpus  --fewshot $fewshot --seed $seed --experiment_config baseline
        # python3 eval_transfer_model.py --src_dataset japflow    --tgt_dataset risec     --fewshot $fewshot --seed $seed --experiment_config baseline
        # python3 eval_transfer_model.py --src_dataset risec      --tgt_dataset japflow   --fewshot $fewshot --seed $seed --experiment_config baseline
        # python3 eval_transfer_model.py --src_dataset japflow    --tgt_dataset mscorpus  --fewshot $fewshot --seed $seed --experiment_config baseline
        # python3 eval_transfer_model.py --src_dataset mscorpus   --tgt_dataset japflow   --fewshot $fewshot --seed $seed --experiment_config baseline

    done
done