#!/bin/bash -e

function CLEAN_AND_EXIT {
    exit $1
}

pushd /opt/dali/docs/examples/use_cases/tensorflow/efficientdet

python -m pip install --upgrade pip
python -m pip install -r requirements.txt


python train.py                                                                                     \
    --epochs 1                                                                                      \
    --input_type coco                                                                               \
    --images_path /data/coco/coco-2017/coco2017/train2017                                           \
    --annotations_path /data/coco/coco-2017/coco2017/annotations/instances_train2017.json           \
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

popd

CLEAN_AND_EXIT 0
