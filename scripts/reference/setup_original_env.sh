#!/usr/bin/env bash
# Python 3.7 environment with the authors' software stack (tensorflow 1.15,
# stable-baselines 2.10.1, gym 0.18), for scripts/reference/run_original_0825a.py.
#
#   bash scripts/reference/setup_original_env.sh <prefix>     # -> <prefix>/bin/python
#
# micromamba is fetched from conda-forge; nothing is installed outside <prefix>.
set -euo pipefail
PREFIX=$(realpath -m "${1:?usage: setup_original_env.sh <prefix>}")
WORK=$(dirname "$PREFIX")/.micromamba
mkdir -p "$WORK"
if [[ ! -x "$WORK/bin/micromamba" ]]; then
  curl -sSfL https://conda.anaconda.org/conda-forge/linux-64/micromamba-1.5.8-0.tar.bz2 \
    | tar -xj -C "$WORK" bin/micromamba
fi
MAMBA_ROOT_PREFIX="$WORK/root" "$WORK/bin/micromamba" create -y -q -p "$PREFIX" -c conda-forge python=3.7 pip
PIP="$PREFIX/bin/pip"
# gym 0.18's setup.py needs an old setuptools
"$PIP" install -q "setuptools==57.5.0" "wheel==0.37.1" "pip<24"
"$PIP" install -q --no-build-isolation tensorflow==1.15.5 stable-baselines==2.10.1 gym==0.18.0 \
  "numpy<1.19" opencv-python-headless==4.5.1.48 matplotlib==3.3.4 pandas==1.1.5 scipy==1.5.4 "protobuf<3.21"
"$PREFIX/bin/python" -c "import tensorflow as tf, stable_baselines, gym; print('tf', tf.__version__, 'sb', stable_baselines.__version__, 'gym', gym.__version__)"
