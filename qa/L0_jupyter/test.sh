#!/bin/bash -e

cd /opt/dali/examples

pip install jupyter numpy==1.11.1 matplotlib python-opencv==3.1.0

mkdir -p /opt/dali/idx_files

ls *.ipynb | xargs -i jupyter nbconvert \
                   --to notebook --execute \
                   --ExecutePreprocessor.kernel_name=python2 \
                   --ExecutePreprocessor.timeout=300 \
                   --output output.ipynb {}
