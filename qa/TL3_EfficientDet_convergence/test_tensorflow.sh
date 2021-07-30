#!/bin/bash -e

function CLEAN_AND_EXIT {
    exit $1
}

pushd /opt/dali/docs/examples/use_cases/tensorflow/efficientdet

python -m pip install --upgrade pip
python -m pip install -r requirements.txt


python train.py \
 --epochs 1 --multi_gpu --seed 1234 --pipeline dali_gpu --input coco \
 --images_path /data/coco/coco-2017/coco2017/train2017 \
 --annotations_path /data/coco/coco-2017/coco2017/annotations/instances_train2017.json \
 --train_batch_size 4 --train_steps 4000 --eval_after_training --eval_during_training \
 --eval_steps 1000 --eval_freq 1 2>&1 | tee $LOG

popd

CLEAN_AND_EXIT 0
