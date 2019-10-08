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
    last_config_index=`./setup_packages.py -n -u $pip_packages --cuda ${CUDA_VERSION}`
    for i in `seq 0 $last_config_index`; do
        eval "packages=(`./setup_packages.py -i $i -u $pip_packages --cuda ${CUDA_VERSION} --include-link | tr -d '[],'`)"
        for pkg in "${packages[@]}"; do
            pip download "$pkg" -d /pip-packages
        done
    done
done
