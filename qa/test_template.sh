topdir=$(cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null && pwd )/..

# Install dependencies
# Note: glib-2.0 depends on python2, so reinstall the desired python afterward to make sure defaults are right
apt-get update
apt-get install -y --no-install-recommends glib-2.0
apt-get install -y --no-install-recommends --reinstall python$PYVER python$PYVER-dev

CUDA_VERSION=$(nvcc --version | grep -E ".*release ([0-9]+)\.([0-9]+).*" | sed 's/.*release \([0-9]\+\)\.\([0-9]\+\).*/\1\2/')
CUDA_VERSION=${CUDA_VERSION:-90}
# Set proper CUDA version for packages, like MXNet, requiring it
pip_packages=$(echo ${pip_packages} | sed "s/##CUDA_VERSION##/${CUDA_VERSION}/")
count=$($topdir/qa/setup_packages.py -n -u $pip_packages)

for i in `seq 0 $count`;
do
    # install pacakges
    inst=$($topdir/qa/setup_packages.py -i $i -u $pip_packages --cuda ${CUDA_VERSION})
    if [ -n "$inst" ]
    then
      pip install $inst
    fi
    # test code
    test_body

    # remove pacakges
    remove=$($topdir/qa/setup_packages.py -r  -u $pip_packages --cuda ${CUDA_VERSION})
    if [ -n "$remove" ]
    then
      pip uninstall -y $remove
    fi
done
