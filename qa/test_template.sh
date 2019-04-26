#!/bin/bash

# Force error checking
set -e
# Force tests to be verbose
set -x

topdir=$(cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null && pwd )/..

# Install dependencies: opencv-python from 3.3.0.10 onwards uses QT which requires
# X11 and other libraries that are not present in clean docker images or bundled there
apt-get update
apt-get install -y --no-install-recommends libsm6 libice6 libxrender1 libxext6 libx11-6 glib-2.0
# Note: glib-2.0 depends on python2, so reinstall the desired python afterward
# to make sure defaults are right
apt-get install -y --no-install-recommends --reinstall python$PYVER python$PYVER-dev

CUDA_VERSION=$(nvcc --version | grep -E ".*release ([0-9]+)\.([0-9]+).*" | sed 's/.*release \([0-9]\+\)\.\([0-9]\+\).*/\1\2/')
CUDA_VERSION=${CUDA_VERSION:-90}
# Set proper CUDA version for packages, like MXNet, requiring it
pip_packages=$(echo ${pip_packages} | sed "s/##CUDA_VERSION##/${CUDA_VERSION}/")
last_config_index=$($topdir/qa/setup_packages.py -n -u $pip_packages --cuda ${CUDA_VERSION})

# Limit to only one configuration (First version of each package)
if [[ $one_config_only = true ]]; then
    echo "Limiting test run to one configuration of packages (first version of each)"
    last_config_index=0
fi

for i in `seq 0 $last_config_index`;
do
    echo "Test run $i"
    # install packages
    inst=$($topdir/qa/setup_packages.py -i $i -u $pip_packages --cuda ${CUDA_VERSION})
    if [ -n "$inst" ]
    then
      pip install $inst
    fi
    # test code
    test_body

    # remove packages
    remove=$($topdir/qa/setup_packages.py -r  -u $pip_packages --cuda ${CUDA_VERSION})
    if [ -n "$remove" ]
    then
      pip uninstall -y $remove
    fi
done
