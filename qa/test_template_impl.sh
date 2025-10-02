#!/bin/bash

# Force error checking
set -e
# Force tests to be verbose
set -x

# make python output unbuffered to get more accurate timestamps
export PYTHONUNBUFFERED=1

topdir=$(cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null && pwd )/..
# supress leaks that are false positive or not related to DALI
export LSAN_OPTIONS=suppressions=$topdir/qa/leak.sup
export ASAN_OPTIONS=start_deactivated=true:detect_leaks=0:detect_container_overflow=0:suppressions=$topdir/qa/address.sup

source $topdir/qa/setup_test_common.sh

# Set runner for python tests
export PYTHONPATH=${PYTHONPATH}:$topdir/qa:$topdir/dali/test/python
python_test_runner_package="nose nose2 nose-timer nose2-test-timer"
# use DALI nose wrapper to patch nose to support Python 3.10
python_test_runner="python -m nose_wrapper"
python_test_args="--verbose --with-timer --timer-top-n 20 -s"
python_invoke_test="${python_test_runner} ${python_test_args}"

# New framework for Python Tests
# During the transition we run both
# When all tests are ported old will be removed
python_new_test_runner="python -m nose2"
python_new_test_args="--verbose --plugin=nose2_test_timer.plugin --with-timer --timer-color --timer-top-n 20"
python_new_invoke_test="${python_new_test_runner} ${python_new_test_args}"

# Set proper CUDA version for packages, like MXNet, requiring it
pip_packages=$(eval "echo \"${pip_packages}\"" | sed "s/##CUDA_VERSION##/${CUDA_VERSION}/")
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

# disable sanitizer as in some OSes it hangs git clone
export OLD_LD_PRELOAD=${LD_PRELOAD}
export LD_PRELOAD=""
source $topdir/qa/setup_dali_extra.sh
export LD_PRELOAD=${OLD_LD_PRELOAD}

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

enable_sanitizer() {
    export ASAN_OPTIONS=symbolize=1:protect_shadow_gap=0:log_path=sanitizer.log:start_deactivated=true:allocator_may_return_null=1:detect_leaks=1:fast_unwind_on_malloc=0:verify_asan_link_order=0:detect_container_overflow=0:suppressions=$topdir/qa/address.sup
    export ASAN_SYMBOLIZER_PATH=$(which llvm-symbolizer)
    # avoid python false positives
    export PYTHONMALLOC=malloc
    # if something calls dlclose on a module that leaks and it happens before asan can extract symbols we get "unknown module"
    # in the stack trace, to prevent this provide dlclose that does nothing
    echo "int dlclose(void* a) { return 0; }" > /tmp/fake_dlclose.c && gcc -shared -o /tmp/libfakeclose.so /tmp/fake_dlclose.c
    export OLD_LD_PRELOAD=${LD_PRELOAD}
    export LD_PRELOAD="/usr/lib/x86_64-linux-gnu/libasan.so /tmp/glibc_fix.so /tmp/libfakeclose.so /usr/lib/x86_64-linux-gnu/libstdc++.so"
    # Workaround for bug in asan ignoring RPATHs https://bugzilla.redhat.com/show_bug.cgi?id=1449604
    export OLD_LD_LIBRARY_PATH2=${LD_LIBRARY_PATH}  # OLD_LD_LIBRARY_PATH variable name already used
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$(python -c 'import nvidia.nvimgcodec as n; import os; print(os.path.dirname(n.__file__))')
}

# turn off sanitizer to avoid breaking any non-related system built-ins
disable_sanitizer() {
    export ASAN_OPTIONS=start_deactivated=true:detect_leaks=0:detect_container_overflow=0:suppressions=$topdir/qa/address.sup
    export LD_PRELOAD=${OLD_LD_PRELOAD}
    export LD_LIBRARY_PATH=${OLD_LD_LIBRARY_PATH2}
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
        grep -q ERROR $topdir/sanitizer.log && exit 1 || true
    fi
    find $topdir -iname "sanitizer.log*" -delete
}

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

# store the original LD_LIBRARY_PATH
OLD_LD_LIBRARY_PATH=$LD_LIBRARY_PATH

for i in `seq 0 $last_config_index`;
do
    echo "Test run $i"
    # seq from 0 to number of elements in `prolog` array - 1
    for variant in $(seq 0 $((${numer_of_prolog_elms}-1))); do
        ${prolog[variant]}
        echo "Test variant run: $variant"
        # install the latest cuda wheel for CUDA 11.x and above tests if it is x86_64
        # or we just want to use CUDA from system, not wheels
        # or we are in conda
        version_ge "${CUDA_VERSION}" "110" && \
          if [ "$(uname -m)" == "x86_64" ] && [ -z "${DO_NOT_INSTALL_CUDA_WHEEL}" ] && [ -z "${CONDA_PREFIX}" ]; then
            NPP_VERSION=$(if [[ $DALI_CUDA_MAJOR_VERSION == "12" ]]; then echo "==12.2.5.30"; else echo ""; fi)
            if [ "$DALI_CUDA_MAJOR_VERSION" -ge 13 ]; then
                install_pip_pkg "pip install --upgrade nvidia-npp~=${DALI_CUDA_MAJOR_VERSION}.0   \
                                                       nvidia-nvjpeg~=${DALI_CUDA_MAJOR_VERSION}.0  \
                                                       nvidia-cufft~=$((DALI_CUDA_MAJOR_VERSION-1)).0 \
                                                       nvidia-nvjpeg2k-cu${DALI_CUDA_MAJOR_VERSION} \
                                                       nvidia-nvtiff-cu${DALI_CUDA_MAJOR_VERSION} \
                                                       -f /pip-packages"
            else
                install_pip_pkg "pip install --upgrade nvidia-npp-cu${DALI_CUDA_MAJOR_VERSION}${NPP_VERSION} \
                                                       nvidia-nvjpeg-cu${DALI_CUDA_MAJOR_VERSION} \
                                                       nvidia-nvjpeg2k-cu${DALI_CUDA_MAJOR_VERSION} \
                                                       nvidia-nvtiff-cu${DALI_CUDA_MAJOR_VERSION} \
                                                       nvidia-cufft-cu${DALI_CUDA_MAJOR_VERSION} \
                                                       -f /pip-packages"
            fi
          fi

        # install packages
        inst=$($topdir/qa/setup_packages.py -i $i -u $pip_packages --cuda ${CUDA_VERSION})
        if [ -n "$inst" ]; then
            for pkg in ${inst}
            do
                install_pip_pkg "pip install $pkg -f /pip-packages ${link_indices_string} ${extra_indices_string}"
            done

            # If we just installed tensorflow, we need to reinstall DALI TF plugin
            if [[ "$inst" == *tensorflow* ]]; then
                # The package name can be nvidia-dali-tf-plugin,  nvidia-dali-tf-plugin-weekly or nvidia-dali-tf-plugin-nightly
                # Different DALI can be installed as a dependency of nvidia-dali so uninstall it too
                pip uninstall -y `pip list | grep nvidia-dali-tf-plugin | cut -d " " -f1` || true
                # don't reinstall DALI wheen in conda as we use conda package
                if [ -z "${CONDA_PREFIX}" ]; then
                    pip uninstall -y `pip list | grep nvidia-dali | cut -d " " -f1` || true
                    pip install /opt/dali/nvidia_dali*.whl;
                fi
                pip install /opt/dali/nvidia_dali_tf_plugin*.tar.gz;
            fi
            # if we are using any cuda or nvidia-tensorflow wheels (nvidia-npp, nvidia-nvjpeg or nvidia-cufft)
            # unset LD_LIBRARY_PATH to not used cuda from /usr/local/ but from wheels
            # however avoid removing compat from the path
            if [[ "$inst" == *nvidia-n* ]] || [[ "$inst" == *nvidia-c* ]] || [[ "$inst" == *nvidia-t* ]]; then
                TAIL=${LD_LIBRARY_PATH#*compat}
                LD_LIBRARY_PATH=${LD_LIBRARY_PATH/$TAIL/}
                export LD_LIBRARY_PATH=${LD_LIBRARY_PATH##*:}
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
        if [ -n "$DALI_ENABLE_SANITIZERS" ]; then
            process_sanitizers_logs
        fi
        # restore the original LD_LIBRARY_PATH
        export LD_LIBRARY_PATH=$OLD_LD_LIBRARY_PATH
        if [ $RV -gt 0 ]; then
            # if sanitizers are enabled don't capture core
            if [ -z "$DALI_ENABLE_SANITIZERS" ]; then
                mkdir -p $topdir/core_artifacts
                cp core* $topdir/core_artifacts || true
            fi
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
