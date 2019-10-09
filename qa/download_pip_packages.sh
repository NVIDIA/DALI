#!/usr/bin/env bash

mkdir -p /pip-packages
gather_pip_packages=yes

for folder in $(ls -d */ | grep -v TL3);
do
    echo ${folder}
    pushd ${folder}
    source test.sh
    popd
    echo ${pip_packages}
    last_config_index=$(python setup_packages.py -n -u $pip_packages --cuda ${CUDA_VERSION})
    for i in `seq 0 $last_config_index`;
    do
        inst=$(python setup_packages.py -i $i -u $pip_packages --cuda ${CUDA_VERSION})
        if [ -n "$inst" ]
        then
            pip download $inst -d /pip-packages
        fi
    done
done
