#!/bin/bash -e

function CLEAN_AND_EXIT {
    exit $1
}

cd /opt/dali/docs/examples/use_cases/tensorflow/yolov4

apt update && apt install python3-opencv -y

python -m pip install --upgrade pip
python -m pip install -r requirements.txt

# turn off SHARP to avoid NCCL errors
export NCCL_NVLS_ENABLE=0
# workaround for https://github.com/tensorflow/tensorflow/issues/63548
export WRAPT_DISABLE_EXTENSIONS=1

python src/main.py train \
    /data/coco/coco-2017/coco2017/train2017 \
    /data/coco/coco-2017/coco2017/annotations/instances_train2017.json \
    -b 2 -e 1 -s 4000 -o output.h5 \
    --learning_rate="1e-3" --pipeline dali-gpu  --multigpu --use_mosaic \
    --eval_frequency 1 --eval_steps 100 \
    --eval_file_root /data/coco/coco-2017/coco2017/val2017 \
    --eval_annotations /data/coco/coco-2017/coco2017/annotations/instances_val2017.json 2>&1 | tee $LOG


CLEAN_AND_EXIT ${PIPESTATUS[0]}
