#!/bin/bash

set -euxo pipefail

apt update

pip install torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric -f https://data.pyg.org/whl/torch-1.12.0+cpu.html
pip install -r /src/requirements.txt