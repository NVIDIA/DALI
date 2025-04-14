#!/bin/bash -e
# used pip packages
pip_packages='numpy'
target_dir=./dali/test/python

do_once() {
    NUM_GPUS=$(nvidia-smi -L | wc -l)
}

test_body() {
    export DATA_DIR=/data/coco/coco-2017/coco2017
    export IS_TMP_DIR=0
    if [ ! -f "/data/coco/coco-2017/coco2017/train2017/000000581929.jpg"] && [ -f "/data/coco/coco-2017/coco2017/train2017.zip"]; then
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
        wait
    fi
    # test code
    python test_data_containers.py --gpus ${NUM_GPUS} -b 2048 -p 10 \
        COCOReaderPipeline="[['${DATA_DIR}/train2017', \
                              '${DATA_DIR}/annotations/instances_train2017.json'], \
                             ['${DATA_DIR}/val2017', \
                              '${DATA_DIR}/annotations/instances_val2017.json']]"
    ((IS_TMP_DIR)) && rm -rf ${DATA_DIR} || true
}

pushd ../..
source ./qa/test_template.sh
popd
