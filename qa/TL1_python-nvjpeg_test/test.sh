#!/bin/bash -e
# used pip packages
pip_packages='${python_test_runner_package} numpy'
target_dir=./dali/test/python


test_body() {
    check_required_paths ${IMAGENET_DIR}/train-jpeg "${COCO_DIR}"
    NUM_GPUS=$(nvidia-smi -L | wc -l)
    if [ $(stat ${IMAGENET_DIR}/train-jpeg --format="%T" -f) != "ext2/ext3" ]; then
        echo "Not available locally, skipping the test"
        return 0
    fi
    prepare_coco_dataset
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
