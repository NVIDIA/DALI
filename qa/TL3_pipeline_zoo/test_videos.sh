#!/bin/bash -ex
pushd /opt/dali/docs/examples/zoo/videos

scripts=(
    "decode.py --videos_dir $DALI_EXTRA_PATH/db/video/sintel/video_files" \
    "decode_and_transform_pytorch.py --videos_dir $DALI_EXTRA_PATH/db/video/sintel/video_files" \
    )
    #TODO: this script requires additional repository to be downloaded
    #"decode_and_transform_from_json.py --videos_dir $DALI_EXTRA_PATH/db/video/sintel/video_files \
for SCRIPT_NAME in "${scripts[@]}"; do
    python $SCRIPT_NAME
    RV=$?
    if [ $RV -gt 0 ]; then
        echo "Failed! $SCRIPT_NAME"
        exit $RV
    fi
 done;

popd

exit 0
