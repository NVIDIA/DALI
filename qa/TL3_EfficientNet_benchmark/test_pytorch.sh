#!/bin/bash -e

# Copyright (c) 2023-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

set -o errexit
set -o pipefail

function CLEAN_AND_EXIT {
    exit $1
}

# enable compat for CUDA 13 if the test image doesn't support it yet
source <(echo "set -x"; cat ../setup_test_common.sh; echo "set +x")

install_cuda_compat

cd /opt/dali/docs/examples/use_cases/pytorch/efficientnet

pip install --no-cache-dir -r requirements.txt

NUM_GPUS=$(nvidia-smi -L | wc -l)

if [ $NUM_GPUS -ne 8 ];
then
    echo "This test requires DGX-1V with 8 GPUs to run correctly"
    exit 1
fi

# Setup /imagenet/{train,val}

mkdir -p /imagenet
pushd /imagenet

if [ ! -d "val" ]; then
   ln -sf /data/imagenet/val-jpeg/ val
fi
if [ ! -d "train" ]; then
   ln -sf /data/imagenet/train-jpeg/ train
fi

popd

# turn off SHARP to avoid NCCL errors
export NCCL_NVLS_ENABLE=0

export PATH_TO_IMAGENET=/imagenet

export RESULT_WORKSPACE=./

# synthetic benchmark
python multiproc.py --nproc_per_node 8 ./main.py --amp --static-loss-scale 128 --batch-size 512 --epochs 1 --prof 1000 --no-checkpoints --training-only --data-backend synthetic --workspace $RESULT_WORKSPACE --report-file bench_report_synthetic.json $PATH_TO_IMAGENET
# -----

# DALI without automatic augmentations
python multiproc.py --nproc_per_node 8 ./main.py --amp --static-loss-scale 128 --batch-size 512 --workers 13 --epochs 3 --no-checkpoints --training-only --data-backend dali --automatic-augmentation disabled --workspace $RESULT_WORKSPACE --report-file bench_report_dali.json $PATH_TO_IMAGENET

# DALI with AutoAugment
python multiproc.py --nproc_per_node 8 ./main.py --amp --static-loss-scale 128 --batch-size 512 --workers 13 --epochs 3 --no-checkpoints --training-only --data-backend dali --automatic-augmentation autoaugment  --workspace $RESULT_WORKSPACE --report-file bench_report_dali_aa.json $PATH_TO_IMAGENET

# DALI with TrivialAugment
python multiproc.py --nproc_per_node 8 ./main.py --amp --static-loss-scale 128 --batch-size 512 --workers 13 --epochs 3 --no-checkpoints --training-only --data-backend dali --automatic-augmentation trivialaugment --workspace $RESULT_WORKSPACE --report-file bench_report_dali_ta.json $PATH_TO_IMAGENET

# DALI proxy without automatic augmentations
python multiproc.py --nproc_per_node 8 ./main.py --amp --static-loss-scale 128 --batch-size 512 --workers 13 --epochs 3 --no-checkpoints --training-only --data-backend dali_proxy --automatic-augmentation disabled  --workspace $RESULT_WORKSPACE --report-file bench_report_dali_proxy.json $PATH_TO_IMAGENET

# DALI proxy with AutoAugment
python multiproc.py --nproc_per_node 8 ./main.py --amp --static-loss-scale 128 --batch-size 512 --workers 13 --epochs 3 --no-checkpoints --training-only --data-backend dali_proxy --automatic-augmentation autoaugment  --workspace $RESULT_WORKSPACE --report-file bench_report_dali_proxy_aa.json $PATH_TO_IMAGENET

# DALI proxy with TrivialAugment
python multiproc.py --nproc_per_node 8 ./main.py --amp --static-loss-scale 128 --batch-size 512 --workers 13 --epochs 3 --no-checkpoints --training-only --data-backend dali_proxy --automatic-augmentation trivialaugment --workspace $RESULT_WORKSPACE --report-file bench_report_dali_proxy_ta.json $PATH_TO_IMAGENET

# PyTorch without automatic augmentations
python multiproc.py --nproc_per_node 8 ./main.py --amp --static-loss-scale 128 --batch-size 512 --workers 10 --epochs 3 --no-checkpoints --training-only --data-backend pytorch --automatic-augmentation disabled --workspace $RESULT_WORKSPACE --report-file bench_report_pytorch.json $PATH_TO_IMAGENET

# PyTorch with AutoAugment:
python multiproc.py --nproc_per_node 8 ./main.py --amp --static-loss-scale 128 --batch-size 512 --workers 10 --epochs 3 --no-checkpoints --training-only --data-backend pytorch --automatic-augmentation autoaugment --workspace $RESULT_WORKSPACE --report-file bench_report_pytorch_aa.json $PATH_TO_IMAGENET

# Optimized PyTorch without automatic augmentations
python multiproc.py --nproc_per_node 8 ./main.py --amp --static-loss-scale 128 --batch-size 512 --workers 10 --epochs 3 --no-checkpoints --training-only --data-backend pytorch_optimized --automatic-augmentation disabled --workspace $RESULT_WORKSPACE --report-file bench_report_optimized_pytorch.json $PATH_TO_IMAGENET

# Optimized PyTorch with AutoAugment:
python multiproc.py --nproc_per_node 8 ./main.py --amp --static-loss-scale 128 --batch-size 512 --workers 10 --epochs 3 --no-checkpoints --training-only --data-backend pytorch_optimized --automatic-augmentation autoaugment --workspace $RESULT_WORKSPACE --report-file bench_report_optimized_pytorch_aa.json $PATH_TO_IMAGENET

# -----

# The line below finds the lines with `train.total_ips`, takes the last one (with the result we
# want) cuts the DLLL (this is highly useful for JSON parsing) from the JSON logs, and parses it
# as JSON using Python. We can now parse the values or directly evaluate the thresholds.
# grep "train.total_ips" <filename>.json | tail -1 | cut -c 5- | python3 -c "import sys, json; print(json.load(sys.stdin))"

# Actual results are about 10% samples/s more
SYNTH_THRESHOLD=38000
DALI_NONE_THRESHOLD=32000
DALI_AA_THRESHOLD=32000
DALI_TA_THRESHOLD=32000
DALI_PROXY_NONE_THRESHOLD=32000
DALI_PROXY_AA_THRESHOLD=32000
DALI_PROXY_TA_THRESHOLD=32000
PYTORCH_NONE_THRESHOLD=32000
PYTORCH_AA_THRESHOLD=32000

function CHECK_PERF_THRESHOLD {
    FILENAME=$1
    THRESHOLD=$2
    grep "train.total_ips" $FILENAME | tail -1 | cut -c 5- | python3 -c "import sys, json
total_ips = json.load(sys.stdin)[\"data\"][\"train.total_ips\"]
if total_ips < $THRESHOLD:
    print(f\"[FAIL] $FILENAME below threshold: {total_ips} < $THRESHOLD\")
    sys.exit(1)
else:
    print(f\"[PASS] $FILENAME above threshold: {total_ips} >= $THRESHOLD\")
    sys.exit(0)"
}


CHECK_PERF_THRESHOLD "bench_report_synthetic.json" $SYNTH_THRESHOLD
CHECK_PERF_THRESHOLD "bench_report_dali.json" $DALI_NONE_THRESHOLD
CHECK_PERF_THRESHOLD "bench_report_dali_aa.json" $DALI_AA_THRESHOLD
CHECK_PERF_THRESHOLD "bench_report_dali_ta.json" $DALI_TA_THRESHOLD
CHECK_PERF_THRESHOLD "bench_report_dali_proxy.json" $DALI_PROXY_NONE_THRESHOLD
CHECK_PERF_THRESHOLD "bench_report_dali_proxy_aa.json" $DALI_PROXY_AA_THRESHOLD
CHECK_PERF_THRESHOLD "bench_report_dali_proxy_ta.json" $DALI_PROXY_TA_THRESHOLD
CHECK_PERF_THRESHOLD "bench_report_pytorch.json" $PYTORCH_NONE_THRESHOLD
CHECK_PERF_THRESHOLD "bench_report_pytorch_aa.json" $PYTORCH_AA_THRESHOLD
CHECK_PERF_THRESHOLD "bench_report_optimized_pytorch.json" $PYTORCH_NONE_THRESHOLD
CHECK_PERF_THRESHOLD "bench_report_optimized_pytorch_aa.json" $PYTORCH_AA_THRESHOLD



# In the initial training we get significant increase in accuracy on the first few epochs,
# after 10 epochs we typically cross 50%.
# Do an additional run of DALI + AA for 10 epochs and check against 48 top1 accuracy (with some
# safety margin).

python multiproc.py --nproc_per_node 8 ./main.py --amp --static-loss-scale 128 --batch-size 128 --epochs 10 --no-checkpoints --data-backend dali --automatic-augmentation autoaugment  --workspace $RESULT_WORKSPACE --report-file accuracy_report_dali_aa.json $PATH_TO_IMAGENET


function CHECK_ACCURACY_THRESHOLD {
    FILENAME=$1
    THRESHOLD=$2
    grep "val.top1" $FILENAME | tail -1 | cut -c 5- | python3 -c "import sys, json
accuracy = json.load(sys.stdin)[\"data\"][\"val.top1\"]
if accuracy < $THRESHOLD:
    print(f\"[FAIL] $FILENAME below threshold: {accuracy} < $THRESHOLD\")
    sys.exit(1)
else:
    print(f\"[PASS] $FILENAME above threshold: {accuracy} >= $THRESHOLD\")
    sys.exit(0)"
}

CHECK_ACCURACY_THRESHOLD "accuracy_report_dali_aa.json" 48
