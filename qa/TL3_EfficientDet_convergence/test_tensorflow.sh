#!/bin/bash -e

function CLEAN_AND_EXIT {
    ((IS_TMP_DIR)) && rm -rf ${DATA_DIR}
    exit $1
}

# enable compat for CUDA 13 if the test image doesn't support it yet
source <(echo "set -x"; cat ../setup_test_common.sh; echo "set +x")

install_cuda_compat

pushd /opt/dali/docs/examples/use_cases/tensorflow/efficientdet

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

python train.py                                                                                     \
    --epochs 1                                                                                      \
    --input_type coco                                                                               \
    --images_path ${DATA_DIR}/train2017                                           \
    --annotations_path ${DATA_DIR}/annotations/instances_train2017.json           \
    --batch_size 3                                                                                  \
    --train_steps 6000                                                                              \
    --eval_steps 1000                                                                               \
    --eval_freq 1                                                                                   \
    --pipeline_type dali_gpu                                                                        \
    --multi_gpu                                                                                     \
    --seed 0                                                                                        \
    --eval_during_training                                                                          \
    --eval_after_training                                                                           \
    --log_dir .                                                                                     \
    --ckpt_dir .                                                                                    \
    --output_filename out_weights_1.h5  2>&1 | tee $LOG

CLEAN_AND_EXIT ${PIPESTATUS[0]}
