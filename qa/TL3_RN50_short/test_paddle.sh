#!/bin/bash -e

set -o errexit
set -o pipefail

function CLEAN_AND_EXIT {
    exit $1
}


# enable compat for CUDA 13 if the test image doesn't support it yet
source <(echo "set -x"; cat ../setup_test_common.sh; echo "set +x")

install_cuda_compat

export USE_CUDA_VERSION=$(echo $(nvcc --version) | sed 's/.*\(release \)\([0-9]\+\)\.\([0-9]\+\).*/\2\3/')
pip install $(python /opt/dali/qa/setup_packages.py -i 0 -u paddlepaddle-gpu --cuda ${USE_CUDA_VERSION})

cd /opt/dali/docs/examples/use_cases/paddle/resnet50

GPUS=$(nvidia-smi -L | sed "s/GPU \([0-9]*\):.*/\1/g")

if [ ! -d "val" ]; then
   ln -sf /data/imagenet/val-jpeg/ val
fi
if [ ! -d "train" ]; then
   ln -sf /data/imagenet/train-jpeg/ train
fi

LOG=dali.log

SECONDS=0
EPOCHS=25  # limiting to 25 epochs to save time

# turn off SHARP to avoid NCCL errors
export NCCL_NVLS_ENABLE=0

export FLAGS_fraction_of_gpu_memory_to_use=.80
python -m paddle.distributed.launch --selected_gpus $(echo $GPUS | tr ' ' ',') \
    main.py -b 96 -j 4 --lr=0.3 --epochs ${EPOCHS} ./ 2>&1 | tee $LOG

RET=${PIPESTATUS[0]}
echo "Training ran in $SECONDS seconds"
if [[ $RET -ne 0 ]]; then
    echo "Error in training script."
    CLEAN_AND_EXIT 2
fi

MIN_TOP1=45.0  # would be 75.0 if we run 90 epochs
MIN_TOP5=70.0  # would be 92.0 if we run 90 epochs
MIN_PERF=2000

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
