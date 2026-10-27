#!/bin/bash -e

function CLEAN_AND_EXIT {
    exit $1
}

# enable compat for CUDA 13 if the test image doesn't support it yet
source <(echo "set -x"; cat ../setup_test_common.sh; echo "set +x")

prepare_coco_dataset

install_cuda_compat

cd /opt/dali/docs/examples/use_cases/tensorflow/yolov4

apt update && apt install python3-opencv -y

python -m pip install --upgrade pip
python -m pip install -r requirements.txt

# turn off SHARP to avoid NCCL errors
export NCCL_NVLS_ENABLE=0
# workaround for https://github.com/tensorflow/tensorflow/issues/63548
export WRAPT_DISABLE_EXTENSIONS=1

python src/main.py train \
    ${DATA_DIR}/train2017 \
    ${DATA_DIR}/annotations/instances_train2017.json \
    -b 2 -e 1 -s 4000 -o output.h5 \
    --learning_rate="1e-3" --pipeline dali-gpu  --multigpu --use_mosaic \
    --eval_frequency 1 --eval_steps 100 \
    --eval_file_root ${DATA_DIR}/val2017 \
    --eval_annotations ${DATA_DIR}/annotations/instances_val2017.json 2>&1 | tee $LOG

CLEAN_AND_EXIT ${PIPESTATUS[0]}
