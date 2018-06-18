#!/bin/bash -e

pushd ../..

# Install dependencies
# Note: glib-2.0 depends on python2, so reinstall the desired python afterward to make sure defaults are right
apt-get update
apt-get install -y --no-install-recommends glib-2.0
apt-get install -y --no-install-recommends --reinstall python$PYVER python$PYVER-dev
pip install jupyter matplotlib opencv-python==3.1.0 mxnet-cu90==1.3.0b20180612 tensorflow-gpu

# attempt to run jupyter on all example notebooks
mkdir -p idx_files
cd examples
find */* -name "*.ipynb" | xargs -i jupyter nbconvert \
                   --to notebook --execute \
                   --ExecutePreprocessor.kernel_name=python2 \
                   --ExecutePreprocessor.timeout=300 \
                   --output output.ipynb {}

popd
