#!/bin/bash -e

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

pip install jupyter matplotlib opencv-python==3.1.0 \
            mxnet-cu90==1.3.0b20180612 \
            tensorflow-gpu \
            http://download.pytorch.org/whl/cu90/torch-0.4.0-${PYVER_TAG}-linux_x86_64.whl \
            torchvision

echo "---------Testing MXNET;DALI----------"
( set -x && python -c "import mxnet; import nvidia.dali.plugin.mxnet" )
echo "---------Testing DALI;MXNET----------"
( set -x && python -c "import nvidia.dali.plugin.mxnet; import mxnet" )

echo "---------Testing TENSORFLOW;DALI----------"
( set -x && python -c "import tensorflow; import nvidia.dali.plugin.tf" )
echo "---------Testing DALI;TENSORFLOW----------"
( set -x && python -c "import nvidia.dali.plugin.tf; import tensorflow" )

echo "---------Testing PYTORCH;DALI----------"
( set -x && python -c "import torch; import nvidia.dali.plugin.pytorch" )
echo "---------Testing DALI;PYTORCH----------"
( set -x && python -c "import nvidia.dali.plugin.pytorch; import torch" )

