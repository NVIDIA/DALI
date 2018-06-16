#!/bin/bash -e

pushd ../..

# Install dependencies
# Note: glib-2.0 depends on python2, so reinstall the desired python afterward to make sure defaults are right
apt-get update
apt-get install -y --no-install-recommends glib-2.0
apt-get install -y --no-install-recommends --reinstall python$PYVER python$PYVER-dev
pip install jupyter numpy==1.11.1 matplotlib opencv-python==3.1.0

# attempt to run jupyter on all example notebooks
mkdir -p idx_files
cd examples
ls *.ipynb | xargs -i jupyter nbconvert \
                   --to notebook --execute \
                   --ExecutePreprocessor.kernel_name=python2 \
                   --ExecutePreprocessor.timeout=300 \
                   --output output.ipynb {}

popd
