#!/bin/bash

set -euxo pipefail

singularity run --nv /home/sgururaj/pytorch_1.12.0-cuda11.3-cudnn8-devel.sif "$@"