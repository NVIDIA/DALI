#!/bin/bash -e

test_body() {
  DIRNAME=$(python -c 'import os; from nvidia import dali; print(os.path.dirname(dali.__file__))')
  # skip non core libs
  for SOFILE in $(find $DIRNAME -not \( -path $DIRNAME/test -prune \) -iname 'libdali*.so' -not -iname '*tf*')
  do
      # first line is for the debug
      echo $SOFILE":"
      nm -gC --defined-only $SOFILE | grep -v "dali::" | grep -i " t " | grep -vx ".*T dali.*" | grep -vx ".*T _fini" | grep -vx ".*T _init" || true
      nm -gC --defined-only $SOFILE | grep -v "dali::" | grep -i " t " | grep -vx ".*T dali.*" | grep -vx ".*T _fini" | grep -vxq ".*T _init" && exit 1
  done

  # Define the expected architectures
  if [ "${DALI_CUDA_MAJOR_VERSION}" == "13" ]; then
      expected_list=$(cat <<EOF
arch = sm_100
arch = sm_110
arch = sm_120
arch = sm_75
arch = sm_80
arch = sm_90
EOF
);
  elif [ "${DALI_CUDA_MAJOR_VERSION}" == "12" ]; then
      if [ "$(uname -p)" == "x86_64" ]; then
          expected_list=$(cat <<EOF
arch = sm_100
arch = sm_50
arch = sm_60
arch = sm_70
arch = sm_80
arch = sm_90
EOF
);
      else
          expected_list=$(cat <<EOF
arch = sm_100
arch = sm_70
arch = sm_80
arch = sm_90
EOF
);
      fi
  fi

  # Get the actual output from cuobjdump, sorted and unique
  # and suppress all errors
  set +e
  actual_list=$(cuobjdump ${DIRNAME}/libdali_kernels.so | grep arch | sort | uniq)
  set -e

  # Compare the two lists
  diff <(echo "$expected_list") <(echo "$actual_list")
  status=$?

  if [ $status -gt 1 ]; then
      echo ${expected_list}
      echo ${actual_list}
      exit 1
  fi

  echo "Done"
}

pushd ../..
source ./qa/test_template.sh
popd
