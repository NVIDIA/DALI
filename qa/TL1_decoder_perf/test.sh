#!/bin/bash -e
# used pip packages
pip_packages='numpy'
target_dir=./internal_tools

LOG1="dali_legacy.log"
LOG2="dali_nvimgcodec.log"
function CLEAN_AND_EXIT {
    rm -rf ${LOG1}
    rm -rf ${LOG2}
    exit $1
}

test_body() {
  # SPEC:DA-11356-002_v04
  if [ "$(uname -p)" == "x86_64" ]; then
    # Hopper
    MIN_PERF=19000;
    # use taskset to avoid inefficient data migration between cores we don't want to use
    taskset --cpu-list 0-127 python hw_decoder_bench.py --width_hint 6000 --height_hint 6000 -b 408 -d 0 -g gpu -w 100 -t 100000 -i ${DALI_EXTRA_PATH}/db/single/jpeg -p rn50 -j 70 --hw_load 0.12 | tee ${LOG1}
    taskset --cpu-list 0-127 python hw_decoder_bench.py --width_hint 6000 --height_hint 6000 -b 408 -d 0 -g gpu -w 100 -t 100000 -i ${DALI_EXTRA_PATH}/db/single/jpeg -p rn50 -j 70 --hw_load 0.12 --experimental_decoder | tee ${LOG2}

  else
    # GraceHopper
    MIN_PERF=29000;
    # use taskset to avoid inefficient data migration between cores we don't want to use
    taskset --cpu-list 0-71 python hw_decoder_bench.py --width_hint 6000 --height_hint 6000 -b 408 -d 0 -g gpu -w 100 -t 100000 -i ${DALI_EXTRA_PATH}/db/single/jpeg -p rn50 -j 72 --hw_load 0.11 | tee ${LOG1}
    taskset --cpu-list 0-71 python hw_decoder_bench.py --width_hint 6000 --height_hint 6000 -b 408 -d 0 -g gpu -w 100 -t 100000 -i ${DALI_EXTRA_PATH}/db/single/jpeg -p rn50 -j 72 --hw_load 0.11 --experimental_decoder | tee ${LOG2}
  fi

  # Regex Explanation:
  # Total Throughput: : Matches the literal string "Total Throughput: ".
  # \K: Resets the start of the match, so anything before \K is not included in the output.
  # [0-9]+(\.[0-9]+)?: Matches the number, with an optional decimal part.
  # (?= frames/sec): ensures " frames/sec" follows the number, but doesn't include it.
  PERF1=$(grep -oP 'Total Throughput: \K[0-9]+(\.[0-9]+)?(?= frames/sec)' ${LOG1})
  PERF2=$(grep -oP 'Total Throughput: \K[0-9]+(\.[0-9]+)?(?= frames/sec)' ${LOG2})

  PERF_RESULT1=$(echo "$PERF1 $MIN_PERF" | awk '{if ($1>=$2) {print "OK"} else { print "FAIL" }}')
  PERF_RESULT2=$(echo "$PERF2 $MIN_PERF" | awk '{if ($1>=$2) {print "OK"} else { print "FAIL" }}')
  # Ensure that PERF2 is no less than 15% smaller than PERF1 (target is 5% max)
  PERF_RESULT3=$(echo "$PERF2 $PERF1" | awk '{if ($1 >= $2 * 0.85) {print "OK"} else { print "FAIL" }}')

  echo "PERF_RESULT1=${PERF_RESULT1}"
  echo "PERF_RESULT2=${PERF_RESULT2}"
  echo "PERF_RESULT3=${PERF_RESULT3}"

  if [[ "$PERF_RESULT1" == "OK" && "$PERF_RESULT2" == "OK" && "$PERF_RESULT3" == "OK" ]]; then
    CLEAN_AND_EXIT 0
  else
    CLEAN_AND_EXIT 4
  fi
}
pushd ../..
source ./qa/test_template.sh
popd
