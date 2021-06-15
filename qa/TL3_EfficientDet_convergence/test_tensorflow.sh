#!/bin/bash -e

function CLEAN_AND_EXIT {
    rm -r train
    rm -r val
    rm output.h5
    rm eval.log
    exit $1
}

cd /opt/dali/docs/examples/use_cases/tensorflow/efficientdet

pip install argparse-utils
pip install absl-py

cd dataset

python create_coco_tfrecord.py \
            --image_dir /data/coco/coco2017/train2017 \
            --object_annotations_file /data/coco/coco2017/annotations/instances_train2017.json \
            --output_file_prefix '../train/train' \
python create_tfrecord_indexes.py \
            --tfrecord_file_pattern '../train/*.tfrecord' \
            --tfrecord2idx_script '/opt/dali/tools/tfrecord2idx' \
    
python create_coco_tfrecord.py \
            --image_dir /data/coco/coco2017/val2017 \
            --object_annotations_file /data/coco/coco2017/annotations/instances_val2017.json \
            --output_file_prefix '../val/val' \
python create_tfrecord_indexes.py \
            --tfrecord_file_pattern '../val/*.tfrecord' \
            --tfrecord2idx_script '/opt/dali/tools/tfrecord2idx' \

cd ..

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
