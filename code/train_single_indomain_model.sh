#! /bin/bash

set -euxo pipefail


set +x
# >>> conda initialize >>>
# !! Contents within this block are managed by 'conda init' !!
__conda_setup="$('/home/sgururaj/miniconda3/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "/home/sgururaj/miniconda3/etc/profile.d/conda.sh" ]; then
        . "/home/sgururaj/miniconda3/etc/profile.d/conda.sh"
    else
        export PATH="/home/sgururaj/miniconda3/bin:$PATH"
    fi
fi
unset __conda_setup
# <<< conda initialize <<<
conda activate amr2
set -x

python train_single_domain_model.py --dataset $1 --seed $2 --experiment_config $3