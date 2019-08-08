#!/bin/bash -e

test_body() {
  DIRNAME=$(python -c 'import os; from nvidia import dali; print(os.path.dirname(dali.__file__))')
  # skip non core libs
  for SOFILE in $(find $DIRNAME -iname 'libdali*.so' -not -iname '*tf*')
  do
      # first line is for the debug
      echo $SOFILE":"
      nm -gC --defined-only $SOFILE | grep -v "dali::" | grep -i " t " | grep -vx ".*T dali.*" | grep -vx ".*T _fini" | grep -vx ".*T _init" || true
      nm -gC --defined-only $SOFILE | grep -v "dali::" | grep -i " t " | grep -vx ".*T dali.*" | grep -vx ".*T _fini" | grep -vxq ".*T _init" && exit 1
  done
  echo "Done"
}

pushd ../..
source ./qa/test_template.sh
popd
