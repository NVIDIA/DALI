#!/bin/bash -ex

test_body() {
  BINNAME=dali_benchmark.bin

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

  "$FULLPATH" --benchmark_filter="RN50*"
}

pushd ../..
source ./qa/test_template.sh
popd
