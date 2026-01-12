#!/bin/bash -e
# used pip packages
pip_packages='numpy'
target_dir=./internal_tools

LOG1="dali_legacy.log"
LOG2="dali_nvimgcodec.log"
LOG1_NDD="dali_ndd_legacy.log"
LOG2_NDD="dali_ndd_nvimgcodec.log"
function CLEAN_AND_EXIT {
    rm -rf ${LOG1}
    rm -rf ${LOG2}
    rm -rf ${LOG1_NDD}
    rm -rf ${LOG2_NDD}
    exit $1
}

test_body() {
  # SPEC:DA-11356-002_v04
  if [ "$(uname -p)" == "x86_64" ]; then
    # Hopper
    MIN_PERF=19000;
    MIN_PERF2=18000;  # TODO(janton): target is to be 19000 as well
    MIN_PERF_NDD=14000;
    MIN_PERF2_NDD=14000;  # TODO(janton): remove this second value.
    # use taskset to avoid inefficient data migration between cores we don't want to use
    taskset --cpu-list 0-127 python hw_decoder_bench.py --width_hint 6000 --height_hint 6000 -b 408 -d 0 -g gpu -w 100 -t 100000 -i ${DALI_EXTRA_PATH}/db/single/jpeg -p rn50 -j 70 --hw_load 0.12 | tee ${LOG1}
    taskset --cpu-list 0-127 python hw_decoder_bench.py --width_hint 6000 --height_hint 6000 -b 408 -d 0 -g gpu -w 100 -t 100000 -i ${DALI_EXTRA_PATH}/db/single/jpeg -p rn50 -j 70 --hw_load 0.12 --experimental_decoder | tee ${LOG2}
    taskset --cpu-list 0-127 python hw_decoder_bench.py --width_hint 6000 --height_hint 6000 -b 408 -d 0 -g gpu -w 100 -t 100000 -i ${DALI_EXTRA_PATH}/db/single/jpeg -p ndd_rn50 -j 70 --hw_load 0.12 | tee ${LOG1_NDD}
    taskset --cpu-list 0-127 python hw_decoder_bench.py --width_hint 6000 --height_hint 6000 -b 408 -d 0 -g gpu -w 100 -t 100000 -i ${DALI_EXTRA_PATH}/db/single/jpeg -p ndd_rn50 -j 70 --hw_load 0.12 --experimental_decoder | tee ${LOG2_NDD}

  else
    # GraceHopper
    MIN_PERF=29000;
    MIN_PERF2=29000;  # TODO(janton): remove this second value.
    MIN_PERF_NDD=21000;
    MIN_PERF2_NDD=21000;  # TODO(janton): remove this second value.
    # use taskset to avoid inefficient data migration between cores we don't want to use
    taskset --cpu-list 0-71 python hw_decoder_bench.py --width_hint 6000 --height_hint 6000 -b 408 -d 0 -g gpu -w 100 -t 100000 -i ${DALI_EXTRA_PATH}/db/single/jpeg -p rn50 -j 72 --hw_load 0.11 | tee ${LOG1}
    taskset --cpu-list 0-71 python hw_decoder_bench.py --width_hint 6000 --height_hint 6000 -b 408 -d 0 -g gpu -w 100 -t 100000 -i ${DALI_EXTRA_PATH}/db/single/jpeg -p rn50 -j 72 --hw_load 0.11 --experimental_decoder | tee ${LOG2}
    taskset --cpu-list 0-71 python hw_decoder_bench.py --width_hint 6000 --height_hint 6000 -b 408 -d 0 -g gpu -w 100 -t 100000 -i ${DALI_EXTRA_PATH}/db/single/jpeg -p ndd_rn50 -j 72 --hw_load 0.11 | tee ${LOG1_NDD}
    taskset --cpu-list 0-71 python hw_decoder_bench.py --width_hint 6000 --height_hint 6000 -b 408 -d 0 -g gpu -w 100 -t 100000 -i ${DALI_EXTRA_PATH}/db/single/jpeg -p ndd_rn50 -j 72 --hw_load 0.11 --experimental_decoder | tee ${LOG2_NDD}
  fi

  # Regex Explanation:
  # Total Throughput: : Matches the literal string "Total Throughput: ".
  # \K: Resets the start of the match, so anything before \K is not included in the output.
  # [0-9]+(\.[0-9]+)?: Matches the number, with an optional decimal part.
  # (?= frames/sec): ensures " frames/sec" follows the number, but doesn't include it.
  extract_perf() {
    log_file="$1"
    grep -oP 'Total Throughput: \K[0-9]+(\.[0-9]+)?(?= frames/sec)' "${log_file}"
  }


  perf_check() {
    #  Checks if the extracted performance value from the specified log file
    # is within a given percentage tolerance of a minimum threshold.

    # Args:
    #   $1: The log file to extract the throughput value from.
    #   $2: The minimum threshold value to compare against.
    #   $3: (Optional) Percent tolerance. If specified, allows value to be
    #       within $2 * (1 - percent/100). Defaults to 0.

    # Returns:
    #   Prints "OK" if value >= min_value*(1-tolerance), "FAIL" otherwise.

    local value=$(extract_perf "$1")
    local min_value=$2
    local percent=${3:-0}
    # Check if value is within percent% of min_value below
    local tolerance=$(awk -v p="$percent" 'BEGIN{print p/100}')
    echo "$value $min_value" | awk -v tol="$tolerance" '{
      lower = $2 * (1 - tol);
      if ($1 >= lower) {print "OK"} else {print "FAIL"}
    }'
  }
  PERF_RESULT1=$(perf_check "${LOG1}" "$MIN_PERF")
  PERF_RESULT2=$(perf_check "${LOG2}" "$MIN_PERF2")
  PERF_RESULT1_NDD=$(perf_check "${LOG1_NDD}" "$MIN_PERF_NDD")
  PERF_RESULT2_NDD=$(perf_check "${LOG2_NDD}" "$MIN_PERF2_NDD")
  PERF_RESULT3=$(perf_check "${LOG2}" "$(extract_perf "${LOG1}")" 5)
  PERF_RESULT3_NDD=$(perf_check "${LOG2_NDD}" "$(extract_perf "${LOG1_NDD}")" 5)

  echo "PERF_RESULT1=${PERF_RESULT1}"
  echo "PERF_RESULT2=${PERF_RESULT2}"
  echo "PERF_RESULT3=${PERF_RESULT3}"
  echo "PERF_RESULT1_NDD=${PERF_RESULT1_NDD}"
  echo "PERF_RESULT2_NDD=${PERF_RESULT2_NDD}"
  echo "PERF_RESULT3_NDD=${PERF_RESULT3_NDD}"

  # if [[ "$PERF_RESULT1" == "OK" && "$PERF_RESULT2" == "OK" && "$PERF_RESULT3" == "OK" && "$PERF_RESULT1_NDD" == "OK"  && "$PERF_RESULT2_NDD" == "OK" && "$PERF_RESULT3_NDD" == "OK" ]]; then
  # don't check experimental decoder performance with dynamic mode
  if [[ "$PERF_RESULT1" == "OK" && "$PERF_RESULT2" == "OK" && "$PERF_RESULT3" == "OK" && "$PERF_RESULT1_NDD" == "OK" ]]; then
    CLEAN_AND_EXIT 0
  else
    CLEAN_AND_EXIT 1
  fi
}
pushd ../..
source ./qa/test_template.sh
popd