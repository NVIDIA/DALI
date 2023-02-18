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

            # get extra index url for given packages - PEP 503 Python Package Index
            extra_indices=$($topdir/qa/setup_packages.py -u $pip_packages --cuda ${CUDA_VERSION} -e)
            extra_indices_string=""
            for e in ${extra_indices}; do
                extra_indices_string="${extra_indices_string} --extra-index-url=${e}"
            done

            # get link index url for given packages -  a URL or path to an html file with links to archives
            link_indices=$($topdir/qa/setup_packages.py -u $pip_packages --cuda ${CUDA_VERSION} -k)
            link_indices_string=""
            for e in ${link_indices}; do
                link_indices_string="${link_indices_string} -f ${e}"
            done

            for i in `seq 0 $last_config_index`;
            do
                inst=$(python ../setup_packages.py -i $i -u $pip_packages --cuda ${CUDA_VERSION})
                if [ -n "$inst" ]
                then
                    pip download $inst -d /pip-packages ${link_indices_string} ${extra_indices_string}
                fi
            done
        fi
    done
    popd
done
