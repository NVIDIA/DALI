#!/bin/bash -e

function CLEAN_AND_EXIT {
    rm -r train
    rm -r val
    rm output.h5
    rm eval.log
    exit $1
}

cd /opt/dali/docs/examples/use_cases/tensorflow/efficientdet

python -m pip install --upgrade pip
python -m pip install -r requirements.txt


python train.py \
 --epochs 1 --multi_gpu --seed 1234 --pipeline dali_gpu --input coco \
 --images_path /data/coco/coco-2017/coco2017/train2017 \
 --annotations_path /data/coco/coco-2017/coco2017/annotations/instances_train2017.json \
 --train_batch_size 4 --train_steps 4000 --eval_after_training --eval_during_training \
 --eval_steps 1000 --eval_freq 1

RET=$(tail -n 1 eval.log | awk '{ exit ($NF < 0.2); }')

CLEAN_AND_EXIT $RET
