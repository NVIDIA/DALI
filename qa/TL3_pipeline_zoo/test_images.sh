#!/bin/bash -ex
pushd /opt/dali/docs/examples/zoo/images

scripts=(
    "decode.py --images_dir $DALI_EXTRA_PATH/db/coco/images/" \
    "decode_and_transform_pytorch.py --landmarks_dir $DALI_EXTRA_PATH/db/face_landmark/" \
    "decode_and_transfrom_from_json.py --coco_dir $DALI_EXTRA_PATH/db/coco/" \
    )

for SCRIPT_NAME in "${scripts[@]}"; do
    if ! eval "python $SCRIPT_NAME"; then
        echo "Failed! $SCRIPT_NAME"
        exit $exit_code
    fi
 done;

popd

exit 0
