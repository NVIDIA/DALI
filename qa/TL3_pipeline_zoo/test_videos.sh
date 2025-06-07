#!/bin/bash -ex
pushd /home/mdabek/workspace/DALI_mdabek_workspace/docs/examples/zoo/videos

scripts=(
    "decode.py --videos_dir $DALI_EXTRA_PATH/db/video/sintel/video_files" \
    "decode_and_transform_pytorch.py --videos_dir $DALI_EXTRA_PATH/db/video/sintel/video_files" \
    )
    #"decode_and_transform_from_json.py --videos_dir $DALI_EXTRA_PATH/db/video/sintel/video_files \
for SCRIPT_NAME in "${scripts[@]}"; do
    if ! eval "python $SCRIPT_NAME"; then
        echo "Failed! $SCRIPT_NAME"
        exit $exit_code
    fi
 done;

popd

exit 0
