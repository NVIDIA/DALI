#!/bin/bash -e
# used pip packages
pip_packages='${python_test_runner_package} numpy'
target_dir=./dali/test/python


test_body() {
    NUM_GPUS=$(nvidia-smi -L | wc -l)
    if [ $(stat /data/imagenet/train-jpeg --format="%T" -f) != "ext2/ext3" ]; then
        echo "Not available locally, skipping the test"
        return 0
    fi
    export DATA_DIR=/data/coco/coco-2017/coco2017
    export IS_TMP_DIR=0
    if [ -f "/data/coco/coco-2017/coco2017/train2017.zip" ]; then
        apt update && apt install -y unzip
        export DATA_DIR=$(mktemp -d)
        export IS_TMP_DIR=1
        pushd ${DATA_DIR}
        cp /data/coco/coco-2017/coco2017/train2017.zip . &
        cp /data/coco/coco-2017/coco2017/val2017.zip . &
        cp /data/coco/coco-2017/coco2017/annotations_trainval2017.zip . &
        wait
        unzip -q train2017.zip &
        unzip -q val2017.zip &
        unzip -q annotations_trainval2017.zip &
        wait
        popd
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
