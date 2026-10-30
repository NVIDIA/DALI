#!/bin/bash -e

function CLEAN_AND_EXIT {
    ((IS_TMP_DIR)) && rm -rf ${DATA_DIR}
    exit $1
}

# enable compat for CUDA 13 if the test image doesn't support it yet
source <(echo "set -x"; cat ../setup_test_common.sh; echo "set +x")

check_required_paths "${COCO_DIR}"

install_cuda_compat

pushd /opt/dali/docs/examples/use_cases/tensorflow/efficientdet

python -m pip install --upgrade pip
python -m pip install -r requirements.txt

# turn off SHARP to avoid NCCL errors
export NCCL_NVLS_ENABLE=0
# workaround for https://github.com/tensorflow/tensorflow/issues/63548
export WRAPT_DISABLE_EXTENSIONS=1

prepare_coco_dataset

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
