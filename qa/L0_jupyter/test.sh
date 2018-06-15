#!/bin/bash -e

cd /opt/dali/examples

pip install jupyter matplotlib

mkdir -p /opt/dali/idx_files

ls *.ipynb | xargs -i jupyter nbconvert \
                   --to notebook --execute \
                   --ExecutePreprocessor.kernel_name=python2 \
                   --ExecutePreprocessor.timeout=300 \
                   --output output.ipynb {}
