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
    --train_file_pattern 'train/*.tfrecord' \
    --train_batch_size 32 \
    --epochs 12 \
    --train_steps 2000 \
    --pipeline dali_gpu \
    --multi_gpu \
    --seed 1234 \
    --hparams 'label_map: "coco" num_classes: 91' \

python eval.py \
    --eval_file_pattern 'val/*.tfrecord' \
    --eval_steps 3000 \
    --pipeline dali_gpu \
    --hparams 'label_map: "coco" num_classes: 91' \
    --weights output.h5 2>&1 | tee eval.log \

RET=$(tail -n 1 eval.log | awk '{ exit ($NF < 0.2); }')

CLEAN_AND_EXIT $RET
