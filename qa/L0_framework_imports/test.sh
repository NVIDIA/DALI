#!/bin/bash -e
# used pip packages
pip_packages="jupyter matplotlib opencv-python mxnet-cu90 tensorflow-gpu torchvision torch"

topdir=$(pwd)/../..
# Install dependencies
# Note: glib-2.0 depends on python2, so reinstall the desired python afterward to make sure defaults are right
apt-get update
apt-get install -y --no-install-recommends glib-2.0
apt-get install -y --no-install-recommends --reinstall python$PYVER python$PYVER-dev

case $PYV in
  ""|"27")
    PYVER_TAG=cp27-cp27mu
    ;;
  *)
    PYVER_TAG=cp${PYV}-cp${PYV}m
    ;;
esac

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
    echo "---------Testing MXNET;DALI----------"
    ( set -x && python -c "import mxnet; import nvidia.dali.plugin.mxnet" )
    echo "---------Testing DALI;MXNET----------"
    ( set -x && python -c "import nvidia.dali.plugin.mxnet; import mxnet" )

    echo "---------Testing TENSORFLOW;DALI----------"
    ( set -x && python -c "import tensorflow; import nvidia.dali.plugin.tf as dali_tf; daliop = dali_tf.DALIIterator()" )
    echo "---------Testing DALI;TENSORFLOW----------"
    ( set -x && python -c "import nvidia.dali.plugin.tf as dali_tf; import tensorflow; daliop = dali_tf.DALIIterator()" )

    echo "---------Testing PYTORCH;DALI----------"
    ( set -x && python -c "import torch; import nvidia.dali.plugin.pytorch" )
    echo "---------Testing DALI;PYTORCH----------"
    ( set -x && python -c "import nvidia.dali.plugin.pytorch; import torch" )
    # remove pacakges
    remove=$($topdir/qa/setup_packages.py -r  -u $pip_packages)
    if [ -n "$remove" ]
    then
      pip uninstall -y $remove
    fi
done 
