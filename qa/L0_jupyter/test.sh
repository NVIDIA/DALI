#!/bin/bash -e
# used pip packages
pip_packages="jupyter numpy matplotlib"

pushd ../..

# attempt to run jupyter on all example notebooks
mkdir -p idx_files

# Apparently gcc/g++ installation is broken in the docker image
if ( ! test `find /usr/lib/gcc -name stddef.h` ); then
    apt-get purge --autoremove -y build-essential g++ gcc libc6-dev
    apt-get update && apt-get install -y build-essential g++ gcc libc6-dev
fi

cd docs/examples

test_body() {
    # test code
    # dummy patern
    black_list_files="#"

    ls *.ipynb | sed "/${black_list_files}/d" | xargs -i jupyter nbconvert \
                    --to notebook --inplace --execute \
                    --ExecutePreprocessor.kernel_name=python${PYVER:0:1} \
                    --ExecutePreprocessor.timeout=300 {}
}

source ../../qa/test_template.sh

popd
