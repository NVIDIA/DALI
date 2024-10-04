#!/bin/bash -e
# used pip packages
pip_packages='numpy'
target_dir=./internal_tools

LOG="dali.log"
function CLEAN_AND_EXIT {
    rm -rf ${LOG}
    exit $1
}

test_body() {
  # SPEC:DA-11356-002_v04
  if [ "$(uname -p)" == "x86_64" ]; then
    # Hopper
    MIN_PERF=19000;
    python hw_decoder_bench.py --width_hint 6000 --height_hint 6000 -b 408 -d 0 -g gpu -w 100 -t 100000 -i ${DALI_EXTRA_PATH}/db/single/jpeg -p rn50 -j 70 --hw_load 0.12 | tee ${LOG}
  else
    # GraceHopper
    MIN_PERF=29000;
    python hw_decoder_bench.py --width_hint 6000 --height_hint 6000 -b 408 -d 0 -g gpu -w 100 -t 100000 -i ${DALI_EXTRA_PATH}/db/single/jpeg -p rn50 -j 72 --hw_load 0.11 | tee ${LOG}
  fi
  PERF=$(grep "fps" ${LOG} | awk '{print $1}')

  PERF_RESULT=$(echo "$PERF $MIN_PERF" | awk '{if ($1>=$2) {print "OK"} else { print "FAIL" }}')
  if [[ "$PERF_RESULT" == "OK" ]]; then
    CLEAN_AND_EXIT 0
  fi

  CLEAN_AND_EXIT 4
}
pushd ../..
source ./qa/test_template.sh
popd
