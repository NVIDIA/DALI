#!/bin/bash

# Force error checking
set -e
# Force tests to be verbose
set -x

topdir=$(cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null && pwd )/..
source $topdir/qa/setup_test_common.sh

# Set proper CUDA version for packages, like MXNet, requiring it
pip_packages=$(echo ${pip_packages} | sed "s/##CUDA_VERSION##/${CUDA_VERSION}/")
last_config_index=$($topdir/qa/setup_packages.py -n -u $pip_packages --cuda ${CUDA_VERSION})

if [ -n "$gather_pip_packages" ]
then
    # early exit
    return 0
fi

source $topdir/qa/setup_dali_extra.sh

target_dir=${target_dir-./}
cd ${target_dir}

# Limit to only one configuration (First version of each package)
if [[ $one_config_only = true ]]; then
    echo "Limiting test run to one configuration of packages (first version of each)"
    last_config_index=0
fi

# some global test setup
if [ "$(type -t do_once)" = 'function' ]; then
    do_once
fi

prolog=${prolog-:}
epilog=${epilog-:}
# get the number of elements in `prolog` array
numer_of_prolog_elms=${#prolog[@]}

for i in `seq 0 $last_config_index`;
do
    echo "Test run $i"
    # seq from 0 to number of elements in `prolog` array - 1
    for variant in $(seq 0 $((${numer_of_prolog_elms}-1))); do
        ${prolog[variant]}
        echo "Test variant run: $variant"
        # install packages
        inst=$($topdir/qa/setup_packages.py -i $i -u $pip_packages --cuda ${CUDA_VERSION})
        if [ -n "$inst" ]
        then
        pip install $inst -f /pip-packages

        # If we just installed tensorflow, we need to reinstall DALI TF plugin
        if [[ "$inst" == *tensorflow-gpu* ]]; then
            pip uninstall -y nvidia-dali-tf-plugin || true
            pip install /opt/dali/nvidia-dali-tf-plugin*.tar.gz
        fi
        fi
        # test code
        test_body

        # remove packages
        remove=$($topdir/qa/setup_packages.py -r  -u $pip_packages --cuda ${CUDA_VERSION})
        if [ -n "$remove" ]
        then
        pip uninstall -y $remove
        fi
        ${epilog[variant]}
    done
done
