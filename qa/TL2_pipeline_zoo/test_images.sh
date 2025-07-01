#!/bin/bash -ex
pip_packages='${python_test_runner_package} pillow numpy torch'
target_dir=./docs/examples/zoo/images
test_body() {
    scripts=(
        "decode.py --images_dir $DALI_EXTRA_PATH/db/coco/images/" \
        "decode_and_transform_pytorch.py --landmarks_dir $DALI_EXTRA_PATH/db/face_landmark/" \
        "decode_and_transfrom_from_json.py --coco_dir $DALI_EXTRA_PATH/db/coco/" \
        )
    for SCRIPT_NAME in "${scripts[@]}"; do
        python $SCRIPT_NAME
        RV=$?
        if [ $RV -gt 0 ]; then
            echo "Failed! $SCRIPT_NAME"
            exit $RV
        fi
     done;
}
pushd ../..
source ./qa/test_template.sh
popd
exit 0
