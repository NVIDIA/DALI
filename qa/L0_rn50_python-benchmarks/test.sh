#!/bin/bash -e
# used pip packages
pip_packages="numpy"

pushd ../..
topdir=$(pwd)

cd dali/benchmark

count=$($topdir/qa/setup_packages.py -n -u $pip_packages)

for i in `seq 0 $count`;
do
    # install pacakges
    inst=$($topdir/qa/setup_packages.py -i $i -u $pip_packages)
    if [ -n "$inst" ]
    then
      pip install $inst
    fi
    # test code
    python resnet50_bench.py
    # remove pacakges
    remove=$($topdir/qa/setup_packages.py -r  -u $pip_packages)
    if [ -n "$remove" ]
    then
      pip uninstall -y $remove
    fi
done 

popd
