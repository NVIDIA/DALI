#!/usr/bin/env bash

mkdir -p /pip-packages
export gather_pip_packages=yes

for folder in $(ls -d */ | grep -v TL3);
do
    echo "Checking folder: " ${folder}
    pushd ${folder}
    # check all test files inside
    for test_file in $(ls -f *.sh);
    do
        export pip_packages=""
        echo "Checking file: " ${test_file}
        source ${test_file}
        echo "PIPs to install: " ${pip_packages}
        if test -n "$pip_packages"
        then
            last_config_index=$(python ../setup_packages.py -n -u $pip_packages --cuda ${CUDA_VERSION})

            # get extra index url for given packages
            extra_indices=$($topdir/qa/setup_packages.py -u $pip_packages --cuda ${CUDA_VERSION} -e)
            extra_indices_string=""
            for e in ${extra_indices}; do
                extra_indices_string="${extra_indices_string} --extra-index-url=${e}"
            done

            for i in `seq 0 $last_config_index`;
            do
                inst=$(python ../setup_packages.py -i $i -u $pip_packages --cuda ${CUDA_VERSION})
                if [ -n "$inst" ]
                then
                    pip download $inst -d /pip-packages ${extra_indices_string}
                fi
            done
        fi
    done
    popd
done
