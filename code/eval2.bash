#!/bin/bash 

python3 eval_transfer_model.py --src_dataset risec --tgt_dataset mscorpus  --fewshot 0.05 --seed 0 --experiment_config dep_residual

python3 eval_transfer_model.py --src_dataset risec --tgt_dataset japflow  --fewshot 0.05 --seed 1 --experiment_config dep_residual

python3 eval_transfer_model.py --src_dataset risec --tgt_dataset japflow  --fewshot 0.2  --seed 2 --experiment_config dep_residual

python3 eval_transfer_model.py --src_dataset mscorpus --tgt_dataset japflow  --fewshot 0.1  --seed 2 --experiment_config dep_residual

python3 eval_transfer_model.py --src_dataset japflow --tgt_dataset mscorpus  --fewshot 0.01  --seed 2 --experiment_config dep_residual

python3 eval_transfer_model.py --src_dataset japflow --tgt_dataset mscorpus  --fewshot 0.1  --seed 0 --experiment_config dep_residual
