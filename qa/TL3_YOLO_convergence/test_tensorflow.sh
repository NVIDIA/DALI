#!/bin/bash -e

function CLEAN_AND_EXIT {
    exit $1
}

# enable compat for CUDA 13 if the test image doesn't support it yet
source <(echo "set -x"; cat ../setup_test_common.sh; echo "set +x")

install_cuda_compat

cd /opt/dali/docs/examples/use_cases/tensorflow/yolov4

apt update && apt install python3-opencv -y

python -m pip install --upgrade pip
python -m pip install -r requirements.txt

# turn off SHARP to avoid NCCL errors
export NCCL_NVLS_ENABLE=0
# workaround for https://github.com/tensorflow/tensorflow/issues/63548
export WRAPT_DISABLE_EXTENSIONS=1

export DATA_DIR=/data/coco/coco-2017/coco2017
export IS_TMP_DIR=0
if [ ! -f "/data/coco/coco-2017/coco2017/train2017/000000581929.jpg" ] && [ -f "/data/coco/coco-2017/coco2017/train2017.zip" ]; then
    export DATA_DIR=$(mktemp -d)
    export IS_TMP_DIR=1
    cd ${DATA_DIR}
    cp /data/coco/coco-2017/coco2017/train2017.zip . &
    cp /data/coco/coco-2017/coco2017/val2017.zip . &
    cp /data/coco/coco-2017/coco2017/annotations_trainval2017.zip . &
    wait
    unzip -q train2017.zip &
    unzip -q val2017.zip &
    unzip -q annotations_trainval2017.zip &
fi

python src/main.py train \
    ${DATA_DIR}/train2017 \
    ${DATA_DIR}/annotations/instances_train2017.json \
    -b 2 -e 1 -s 4000 -o output.h5 \
    --learning_rate="1e-3" --pipeline dali-gpu  --multigpu --use_mosaic \
    --eval_frequency 1 --eval_steps 100 \
    --eval_file_root ${DATA_DIR}/val2017 \
    --eval_annotations ${DATA_DIR}/annotations/instances_val2017.json 2>&1 | tee $LOG


CLEAN_AND_EXIT ${PIPESTATUS[0]}
