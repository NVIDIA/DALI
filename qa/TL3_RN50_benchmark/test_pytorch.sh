#!/bin/bash -e

set -o errexit
set -o pipefail

# enable compat for CUDA 13 if the test image doesn't support it yet
source <(echo "set -x"; cat ../setup_test_common.sh; echo "set +x")

install_cuda_compat

cd /opt/dali/docs/examples/use_cases/pytorch/resnet50

NUM_GPUS=$(nvidia-smi -L | wc -l)

if [ ! -d "val" ]; then
   ln -sf /data/imagenet/val-jpeg/ val
fi
if [ ! -d "train" ]; then
   ln -sf /data/imagenet/train-jpeg/ train
fi

# turn off SHARP to avoid NCCL errors
export NCCL_NVLS_ENABLE=0

# Function to check the training results from a log file
check_training_results() {
    local LOG="$1"

    RET=${PIPESTATUS[0]}
    if [[ $RET -ne 0 ]]; then
        echo "Error in training script."
        return 2
    fi

    # Define the minimum performance thresholds
    local MIN_TOP1=15.0
    local MIN_TOP5=35.0
    local MIN_PERF=2900

    # Extract relevant information from the log file
    local TOP1=$(grep "^##Top-1" "$LOG" | awk '{print $2}')
    local TOP5=$(grep "^##Top-5" "$LOG" | awk '{print $2}')
    local PERF=$(grep "^##Perf" "$LOG" | awk '{print $2}')

    # Check if the TOP1 and TOP5 values are available
    if [[ -z "$TOP1" || -z "$TOP5" ]]; then
        echo "Incomplete output."
        return 3
    fi

    # Compare results against the minimum thresholds
    local TOP1_RESULT=$(echo "$TOP1 $MIN_TOP1" | awk '{if ($1>=$2) {print "OK"} else { print "FAIL" }}')
    local TOP5_RESULT=$(echo "$TOP5 $MIN_TOP5" | awk '{if ($1>=$2) {print "OK"} else { print "FAIL" }}')
    local PERF_RESULT=$(echo "$PERF $MIN_PERF" | awk '{if ($1>=$2) {print "OK"} else { print "FAIL" }}')

    # Display results
    echo
    printf "TOP-1 Accuracy: %.2f%% (expect at least %f%%) %s\n" $TOP1 $MIN_TOP1 $TOP1_RESULT
    printf "TOP-5 Accuracy: %.2f%% (expect at least %f%%) %s\n" $TOP5 $MIN_TOP5 $TOP5_RESULT
    printf "Average perf: %.2f (expect at least %f) samples/sec %s\n" $PERF $MIN_PERF $PERF_RESULT

    # If all results are "OK", exit with status 0
    if [[ "$TOP1_RESULT" == "OK" && "$TOP5_RESULT" == "OK" && "$PERF_RESULT" == "OK" ]]; then
        return 0
    fi
    return 4
}

torchrun --nproc_per_node=${NUM_GPUS} main.py -a resnet50 --b 256 --loss-scale 128.0 --workers 8 --lr=0.4 --fp16-mode --epochs 5 --data_loader dali ./ 2>&1 | tee dali.log
check_training_results dali.log
RESULT_DALI=$?

torchrun --nproc_per_node=${NUM_GPUS} main.py -a resnet50 --b 256 --loss-scale 128.0 --workers 8 --lr=0.4 --fp16-mode --epochs 5 --data_loader dali_proxy ./ 2>&1 | tee dali_proxy.log
check_training_results dali_proxy.log
RESULT_DALI_PROXY=$?

# Return 0 if both are 0, otherwise return the first non-zero code
exit ${RESULT_DALI:-$RESULT_DALI_PROXY}
