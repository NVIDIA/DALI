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

install_pip_pkg() {
    install_cmd="$@"
    # if no package was found in our download dir, so install it from index
    ${install_cmd} --no-index || ${install_cmd}
}

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
    last_config_index=$(( 0 > $last_config_index ? $last_config_index : 0 ))
fi

# some global test setup
if [ "$(type -t do_once)" = 'function' ]; then
    do_once
fi

prolog=${prolog-:}
epilog=${epilog-:}
# get the number of elements in `prolog` array
numer_of_prolog_elms=${#prolog[@]}

# turn on sanitizers
enable_sanitizer() {
    # supress leaks that are false positive or not related to DALI
    export LSAN_OPTIONS=suppressions=$topdir/qa/leak.sup
    export ASAN_OPTIONS=symbolize=1:protect_shadow_gap=0:log_path=sanitizer.log:start_deactivated=true:allocator_may_return_null=1::detect_leaks=1
    export ASAN_SYMBOLIZER_PATH=$(which llvm-symbolizer)
    # avoid python false positives
    export PYTHONMALLOC=malloc
}

# turn off sanitizer to avoid breaking any non-related system built-ins
disable_sanitizer() {
    export ASAN_OPTIONS=start_deactivated=true:detect_leaks=0
    unset ASAN_SYMBOLIZER_PATH
    unset PYTHONMALLOC
}

# Wrap the test_body in a subshell, where we can safely execute it with `set -e`
# and turn it off in current shell to intercept the error code
# when sanitizers are on, do set +e to run all the test no matter what the result is
# and collect as much of sanitizers output as possible
test_body_wrapper() {(
    if [ -n "$DALI_ENABLE_SANITIZERS" ]; then
        set +e
        enable_sanitizer
    else
        set -e
    fi

    test_body

    if [ -n "$DALI_ENABLE_SANITIZERS" ]; then
        disable_sanitizer
    fi
)}

process_sanitizers_logs() {
    find $topdir -iname "sanitizer.log.*" -print0 | xargs -0 -I file cat file > $topdir/sanitizer.log
    if [ -e $topdir/sanitizer.log ]; then
        cat $topdir/sanitizer.log
        grep -q ERROR $topdir/sanitizer.log || true
        # ToDo - enable when the suppression file is completed
        # grep -q ERROR $topdir/sanitizer.log && exit 1 || true
    fi
    # rm so the consequitive test won't reread the same logs over and over
    find $topdir -iname "sanitizer.log.*" -delete
}

# get extra index url for given packages
extra_indices=$($topdir/qa/setup_packages.py -u $pip_packages --cuda ${CUDA_VERSION} -e)
extra_indices_string=""
for e in ${extra_indices}; do
    extra_indices_string="${extra_indices_string} --extra-index-url=${e}"
done

for i in `seq 0 $last_config_index`;
do
    echo "Test run $i"
    # seq from 0 to number of elements in `prolog` array - 1
    for variant in $(seq 0 $((${numer_of_prolog_elms}-1))); do
        ${prolog[variant]}
        echo "Test variant run: $variant"
        # install packages
        inst=$($topdir/qa/setup_packages.py -i $i -u $pip_packages --cuda ${CUDA_VERSION})
        if [ -n "$inst" ]; then
            for pkg in ${inst}
            do
                install_pip_pkg "pip install $pkg -f /pip-packages ${extra_indices_string}"
            done

            # If we just installed tensorflow, we need to reinstall DALI TF plugin
            if [[ "$inst" == *tensorflow* ]]; then
                # The package name can be nvidia-dali-tf-plugin,  nvidia-dali-tf-plugin-weekly or nvidia-dali-tf-plugin-nightly
                # Different DALI can be installed as a dependency of nvidia-dali so uninstall it too
                pip uninstall -y `pip list | grep nvidia-dali-tf-plugin | cut -d " " -f1` || true
                pip uninstall -y `pip list | grep nvidia-dali | cut -d " " -f1` || true
                pip install /opt/dali/nvidia_dali*.whl
                pip install /opt/dali/nvidia-dali-tf-plugin*.tar.gz
            fi
        fi
        # test code
        # Run test_body in subshell, the exit on error is turned off in current shell,
        # but it will be present in subshell (thanks to wrapper).
        # We can intercept first error that happens. test_body_wrapper cannot be used with
        # any conditional as it will turn on "exit on error" behaviour
        set +e
        test_body_wrapper
        RV=$?
        set -e
        # if sanitizers are enabled run test until the end so we have as much data as possible
        if [ $RV -gt 0 ] && [ -z "$DALI_ENABLE_SANITIZERS" ]; then
            mkdir -p $topdir/core_artifacts
            cp core* $topdir/core_artifacts || true
            exit ${RV}
        fi
        # remove packages
        remove=$($topdir/qa/setup_packages.py -r -u $pip_packages --cuda ${CUDA_VERSION})
        if [ -n "$remove" ]; then
           pip uninstall -y $remove
        fi
        ${epilog[variant]}
    done
done

if [ -n "$DALI_ENABLE_SANITIZERS" ]; then
    process_sanitizers_logs
fi
