#!/bin/bash

set -euxo pipefail

for ((i=1;i<=$2;i++))
do  
    sbatch --partition=shire_general --gres=gpu:A4500:1 ./singularity_runner.sh ./wandb_agent.sh $1
done