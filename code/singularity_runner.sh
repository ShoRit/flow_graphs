#!/bin/bash

set -euxo pipefail

singularity run --nv ~/pytorch_1.12.0-cuda11.3-cudnn8-devel.sif "$@"