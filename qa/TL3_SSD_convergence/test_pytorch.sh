#!/bin/bash -e

set -o nounset
set -o errexit
set -o pipefail

function CLEAN_AND_EXIT {
    exit $1
}

cd /opt/dali/docs/examples/use_cases/pytorch/single_stage_detector/

pip install mlperf_compliance Cython==0.28.4
pip install pycocotools==2.0.0

NUM_GPUS=$(nvidia-smi -L | wc -l)

LOG=dali.log

SECONDS=0
python -m torch.distributed.launch --nproc_per_node=${NUM_GPUS} main.py --backbone resnet50 --warmup 300 --bs 64 --eval-batch-size 8 --data /coco --data /data/coco/coco-2017/coco2017/ --data_pipeline dali --target 0.25 2>&1 | tee $LOG

RET=${PIPESTATUS[0]}
echo "Training ran in $SECONDS seconds"
if [[ $RET -ne 0 ]]; then
    echo "Error in training script."
    CLEAN_AND_EXIT 2
fi

CLEAN_AND_EXIT 0
