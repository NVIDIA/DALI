#!/bin/bash -e

pip_packages="nose numpy"

apt-get update
apt-get install -y wget ffmpeg

pushd ../..

source qa/setup_dali_extra.sh

cd docs/examples/video

mkdir -p video_files

container_path=${DALI_EXTRA_PATH}/db/optical_flow/sintel_trailer/sintel_trailer.mp4

IFS='/' read -a container_name <<< "$container_path"
IFS='.' read -a splitted <<< "${container_name[-1]}"

for i in {0..4};
do
    ffmpeg -ss 00:00:${i}0 -t 00:00:10 -i $container_path -vcodec copy -acodec copy -y video_files/${splitted[0]}_$i.${splitted[1]}
done


test_body() {
    # test code
    # First running simple code
    python video_example.py

    nosetests --verbose ../../../dali/test/python/test_video_pipeline.py
}

source ../../../qa/test_template.sh

popd
