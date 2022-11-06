#!/bin/bash 

set -euxo pipefail # fail for each command

singularity run --nv /home/rdutt/pytorch_1.12.0-cuda11.3-cudnn8-devel.sif /projects/flow_graphs/code/eval.sh