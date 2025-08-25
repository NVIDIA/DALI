#!/bin/bash -e

set -o errexit
set -o pipefail

function CLEAN_AND_EXIT {
    exit $1
}

# enable compat for CUDA 13 if the test image doesn't support it yet
source <(echo "set -x"; cat ../setup_test_common.sh; echo "set +x")

install_cuda_compat

cd /opt/dali/docs/examples/use_cases/pytorch/resnet50

NUM_GPUS=$(nvidia-smi -L | wc -l)

# turn off SHARP to avoid NCCL errors
export NCCL_NVLS_ENABLE=0

if [ ! -d "val" ]; then
   ln -sf /data/imagenet/val-jpeg/ val
fi
if [ ! -d "train" ]; then
   ln -sf /data/imagenet/train-jpeg/ train
fi

LOG=dali.log

SECONDS=0
torchrun --nproc_per_node=${NUM_GPUS} main.py -a resnet50 --dali_cpu --b 128 --loss-scale 128.0 --workers 4 --lr=0.4 --fp16-mode ./ 2>&1 | tee $LOG

RET=${PIPESTATUS[0]}
echo "Training ran in $SECONDS seconds"
if [[ $RET -ne 0 ]]; then
    echo "Error in training script."
    CLEAN_AND_EXIT 2
fi

MIN_TOP1=75.0
MIN_TOP5=92.0
MIN_PERF=13000

TOP1=$(grep "^##Top-1" $LOG | awk '{print $2}')
TOP5=$(grep "^##Top-5" $LOG | awk '{print $2}')
PERF=$(grep "^##Perf" $LOG | awk '{print $2}')

if [[ -z "$TOP1" || -z "$TOP5" ]]; then
    echo "Incomplete output."
    CLEAN_AND_EXIT 3
fi

TOP1_RESULT=$(echo "$TOP1 $MIN_TOP1" | awk '{if ($1>=$2) {print "OK"} else { print "FAIL" }}')
TOP5_RESULT=$(echo "$TOP5 $MIN_TOP5" | awk '{if ($1>=$2) {print "OK"} else { print "FAIL" }}')
PERF_RESULT=$(echo "$PERF $MIN_PERF" | awk '{if ($1>=$2) {print "OK"} else { print "FAIL" }}')

echo
printf "TOP-1 Accuracy: %.2f%% (expect at least %f%%) %s\n" $TOP1 $MIN_TOP1 $TOP1_RESULT
printf "TOP-5 Accuracy: %.2f%% (expect at least %f%%) %s\n" $TOP5 $MIN_TOP5 $TOP5_RESULT
printf "Average perf: %.2f (expect at least %f) samples/sec %s\n" $PERF $MIN_PERF $PERF_RESULT

if [[ "$TOP1_RESULT" == "OK" && "$TOP5_RESULT" == "OK" && "$PERF_RESULT" == "OK" ]]; then
    CLEAN_AND_EXIT 0
fi

CLEAN_AND_EXIT 4
