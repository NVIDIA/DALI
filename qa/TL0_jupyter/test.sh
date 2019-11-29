#!/bin/bash -e
# used pip packages
pip_packages="jupyter numpy matplotlib pillow opencv-python"
if [ "$PYVER" != "2.7" ]; then
    pip_packages="${pip_packages} simpleaudio" 
fi
target_dir=./docs/examples

do_once() {
    apt update
    apt install -y libasound2-dev
    # attempt to run jupyter on all example notebooks
    mkdir -p idx_files
}

test_body() {

    # test code
    # dummy patern
    black_list_files="multigpu"

    # Blacklist for python2. Can be removed after dropping python2
    if [ "$PYVER" == "2.7" ]; then
        black_list_files="multigpu\|audiodecoder"
    fi    

    ls *.ipynb | sed "/${black_list_files}/d" | xargs -i jupyter nbconvert \
                    --to notebook --inplace --execute \
                    --ExecutePreprocessor.kernel_name=python${PYVER:0:1} \
                    --ExecutePreprocessor.timeout=300 {}
}

pushd ../..
source ./qa/test_template.sh
popd
