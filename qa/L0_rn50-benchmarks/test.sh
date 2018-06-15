#!/bin/bash -ex

BINNAME=dali_benchmark.bin

for DIRNAME in \
  "../../build/ndll/python/dali" \
  "../../build-*$PYV*/ndll/python/dali" \
  "$(python -c 'import os; import dali; print os.path.dirname(dali.__file__)' 2>/dev/null || echo '')"
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

