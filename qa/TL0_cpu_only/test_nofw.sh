#!/bin/bash -e
# used pip packages
pip_packages='${python_test_runner_package} numpy numba scipy librosa'

target_dir=./dali/test/python

test_body() {
  # CPU only test, remove CUDA from the search path just in case
  if [ -n "$DALI_ENABLE_SANITIZERS" ]; then
    # ASAN ignores DT_RUNPATH (https://bugzilla.redhat.com/show_bug.cgi?id=1449604),
    # so libdali_operators.so's lazy dlopen("libnvimgcodec.so") cannot rely on its
    # $ORIGIN/../nvimgcodec rpath entry. Keep the wheel-installed nvimgcodec dir
    # reachable via LD_LIBRARY_PATH instead. Mirrors the same workaround in
    # qa/test_template_impl.sh::enable_sanitizer().
    export LD_LIBRARY_PATH="$(python -c 'import nvidia.nvimgcodec as n, os; print(os.path.dirname(n.__file__))' 2>/dev/null || echo '')"
  else
    export LD_LIBRARY_PATH=""
  fi
  export PATH=${PATH/cuda/}

  for BINNAME in \
    "dali_core_test.bin" \
    "dali_kernel_test.bin" \
    "dali_test.bin" \
    "dali_operator_test.bin"
  do
    for DIRNAME in \
      "../../build/dali/python/nvidia/dali" \
      "$(python -c 'import os; from nvidia import dali; print(os.path.dirname(dali.__file__))' 2>/dev/null || echo '')"
    do
        if [ -x "$DIRNAME/test/$BINNAME" ]; then
            FULLPATH="$DIRNAME/test/$BINNAME"
            break
        fi
    done

    if [[ -z "$FULLPATH" ]]; then
        echo "ERROR: $BINNAME not found"
        exit 1
    fi

    "$FULLPATH" --gtest_filter="*CpuOnly*:*CApi*/0.*-*0.UseCopyKernel:*ForceNoCopyFail:*daliOutputCopySamples"
  done
  if [ -z "$DALI_ENABLE_SANITIZERS" ]; then
    ${python_new_invoke_test} -A '!pytorch' test_dali_cpu_only
  else
    ${python_new_invoke_test} -A '!pytorch,!numba' test_dali_cpu_only
  fi
}

pushd ../..
source ./qa/test_template.sh
popd
