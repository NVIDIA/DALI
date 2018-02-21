#!/bin/bash
set -e

cd /opt/ndll/examples

pip install jupyter matplotlib

mkdir /opt/ndll/idx_files

ls *.ipynb | xargs -i jupyter nbconvert \
                   --to notebook --execute \
                   --ExecutePreprocessor.kernel_name=python2 \
                   --ExecutePreprocessor.timeout=300 \
                   --output output.ipynb {}

rm output.ipynb
rm -r /opt/ndll/idx_files
