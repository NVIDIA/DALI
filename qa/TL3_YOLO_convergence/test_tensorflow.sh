#!/bin/bash -e

function CLEAN_AND_EXIT {
    rm dali.log
    rm output.h5
    exit $1
}

cd /opt/dali/docs/examples/use_cases/tensorflow/yolov4/src

pip install pycocotools==2.0.0

python main.py train \
    /data/coco/coco-2017/coco2017/train2017 \
    /data/coco/coco-2017/coco2017/annotations/instances_train2017.json \
    -b 8 -e 12 -s 3000 -o output.h5 \
    --learning_rate="1e-3" --pipeline dali-gpu  --multigpu --use_mosaic

python main.py eval \
    /data/coco/coco-2017/coco2017/val2017/ \
    /data/coco/coco-2017/coco2017/annotations/instances_val2017.json \
    -b 1 -s 5000 --weights output.h5 2>&1 | tee dali.log

RET=$(tail -n 1 dali.log | awk '{ exit ($NF < 0.2); }')

CLEAN_AND_EXIT $RET
