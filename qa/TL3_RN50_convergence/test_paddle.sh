#!/bin/bash -e

set -o errexit
set -o pipefail

function CLEAN_AND_EXIT {
    exit $1
}

# enable compat for CUDA 13 if the test image doesn't support it yet
source <(echo "set -x"; cat ../setup_test_common.sh; echo "set +x")

install_cuda_compat

cd /opt/dali/docs/examples/use_cases/paddle/resnet50

pip install --no-cache-dir -r requirements.txt

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
export FLAGS_apply_pass_to_program=1

python -m paddle.distributed.launch --gpus=$(echo $GPUS | tr ' ' ',') train.py \
    --epochs ${EPOCHS} \
    --batch-size 96 \
    --amp \
    --scale-loss 128.0 \
    --dali-num-threads 4 \
    --use-dynamic-loss-scaling \
    --data-layout NHWC \
    --report-file train.json \
    --image-root ./ 2>&1 | tee $LOG

RET=${PIPESTATUS[0]}
echo "Training ran in $SECONDS seconds"
if [[ $RET -ne 0 ]]; then
    echo "Error in training script."
    CLEAN_AND_EXIT 2
fi

MIN_TOP1=.45  # would be 75% if we run 90 epochs
MIN_TOP5=.70  # would be 92% if we run 90 epochs
MIN_PERF=27000

function PRINT_THRESHOLD {
    FILENAME=$1
    QUALITY=$2
    THRESHOLD=$3
    grep "$QUALITY" $FILENAME | tail -1 | cut -c 5- | python3 -c "import sys, json
value = json.load(sys.stdin)[\"data\"][\"$QUALITY\"]
print(f\"$FILENAME: $QUALITY: {value}, expected $THRESHOLD\")
sys.exit(0)"
}

function CHECK_THRESHOLD {
    FILENAME=$1
    QUALITY=$2
    THRESHOLD=$3
    grep "$QUALITY" $FILENAME | tail -1 | cut -c 5- | python3 -c "import sys, json
value = json.load(sys.stdin)[\"data\"][\"$QUALITY\"]
if value < $THRESHOLD:
    print(f\"[FAIL] $FILENAME below threshold: {value} < $THRESHOLD\")
    sys.exit(1)
else:
    print(f\"[PASS] $FILENAME above threshold: {value} >= $THRESHOLD\")
    sys.exit(0)"
}

PRINT_THRESHOLD "train.json" "train.ips" $MIN_PERF
PRINT_THRESHOLD "train.json" "val.top1" $MIN_TOP1
PRINT_THRESHOLD "train.json" "val.top5" $MIN_TOP5
CHECK_THRESHOLD "train.json" "train.ips" $MIN_PERF
CHECK_THRESHOLD "train.json" "val.top1" $MIN_TOP1
CHECK_THRESHOLD "train.json" "val.top5" $MIN_TOP5

CLEAN_AND_EXIT 0
