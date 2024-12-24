#!/bin/bash -e

set -o nounset
set -o errexit
set -o pipefail

cd /opt/dali/docs/examples/use_cases/pytorch/resnet50

NUM_GPUS=$(nvidia-smi -L | wc -l)

# turn off SHARP to avoid NCCL errors
export NCCL_NVLS_ENABLE=0

if [ ! -d "/data/imagenet/val-jpeg" ]; then
    echo "Error: /data/imagenet/val-jpeg directory not found."
    exit 1
fi
ln -sf /data/imagenet/val-jpeg/ val

if [ ! -d "/data/imagenet/train-jpeg" ]; then
    echo "Error: /data/imagenet/train-jpeg directory not found."
    exit 1
fi
ln -sf /data/imagenet/train-jpeg/ train

# Function to wrap a full command line
CHECK_PERF_THRESHOLD() {
  local cmd="$1"  # The full command line as a single argument
  shift           # Shift removes the command itself from the arguments list
  local log_filename="$1"
  local fps_threshold="$2"
  declare -n PERF="$3"  # Create a reference to the variable passed

  # Validate arguments
  if [[ -z "$cmd" || -z "$log_filename" || -z "$fps_threshold" ]]; then
    echo "Usage: CHECK_PERF_THRESHOLD '<command> <args>' <log_filename> <fps_threshold>"
    exit 1
  fi

  # Execute the command
  eval "$cmd" ./ 2>&1 | tee "$log_filename"

  RET=${PIPESTATUS[0]}
  if [[ $RET -ne 0 ]]; then
    echo "Error: Command failed"
    exit 1
  fi

  echo "Command executed successfully. Output saved to $log_filename."

  TOP1=$(grep "^##Top-1" $log_filename | awk '{print $2}')
  TOP5=$(grep "^##Top-5" $log_filename | awk '{print $2}')
  PERF=$(grep "^##Perf" $log_filename | awk '{print $2}')

  PERF_RESULT=$(echo "$PERF $fps_threshold" | awk '{if ($1>=$2) {print "OK"} else { print "FAIL" }}')
  printf "Average perf: %.2f (expect at least %f) samples/sec %s\n" $PERF $fps_threshold $PERF_RESULT

  if [[ "$PERF_RESULT" == "FAIL" ]]; then
    exit 1
  fi
}

# ref < dali_proxy < dali
CHECK_PERF_THRESHOLD "torchrun --nproc_per_node=${NUM_GPUS} main.py -a resnet50 --b 256 --loss-scale 128.0 --disable_dali --workers 4 --lr=0.4 --fp16-mode --epochs 2" "dali0.log" "0" "REF_THROUGHPUT"
CHECK_PERF_THRESHOLD "torchrun --nproc_per_node=${NUM_GPUS} main.py -a resnet50 --b 256 --loss-scale 128.0 --dali_proxy --workers 4 --lr=0.4 --fp16-mode --epochs 2" "dali1.log" "${REF_THROUGHPUT}" "PERF_PROXY"
CHECK_PERF_THRESHOLD "torchrun --nproc_per_node=${NUM_GPUS} main.py -a resnet50 --b 256 --loss-scale 128.0 --workers 4 --lr=0.4 --fp16-mode --epochs 2" "dali2.log" "${DALI_PROXY_THROUGHPUT}" "PERF_DALI"

CHECK_PERF_THRESHOLD "torchrun --nproc_per_node=${NUM_GPUS} main.py -a resnet50 --b 128 --loss-scale 128.0 --disable_dali --workers 4 --lr=0.4 --fp16-mode --epochs 2" "dali3.log" "0" "REF_THROUGHPUT2"
CHECK_PERF_THRESHOLD "torchrun --nproc_per_node=${NUM_GPUS} main.py -a resnet50 --dali_cpu --b 128 --loss-scale 128.0 --dali_proxy --workers 4 --lr=0.4 --fp16-mode --epochs 2" "dali4.log" "${REF_THROUGHPUT2}" "PERF_PROXY"
CHECK_PERF_THRESHOLD "torchrun --nproc_per_node=${NUM_GPUS} main.py -a resnet50 --dali_cpu --b 128 --loss-scale 128.0 --workers 4 --lr=0.4 --fp16-mode --epochs 2" "dali5.log" "${DALI_PROXY_THROUGHPUT2}" "PERF_DAL2"

exit 0