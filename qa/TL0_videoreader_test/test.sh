#!/bin/bash -e

pip_packages="nose numpy"

apt-get update
apt-get install -y wget ffmpeg

pushd ../..

source qa/setup_dali_extra.sh

cd docs/examples/video

mkdir -p video_files
mkdir -p labelled_videos/{0..2}

container_path=${DALI_EXTRA_PATH}/db/optical_flow/sintel_trailer/sintel_trailer.mp4

IFS='/' read -a container_name <<< "$container_path"
IFS='.' read -a split <<< "${container_name[-1]}"

for i in {0..4};
do
    ffmpeg -ss 00:00:${i}0 -t 00:00:10 -i $container_path -vcodec copy -acodec copy -y video_files/${split[0]}_$i.${split[1]}
done

for i in {0..9};
do
    ffmpeg -ss 00:00:$((i*5)) -t 00:00:05 -i $container_path -vcodec copy -acodec copy -y labelled_videos/$((i % 3))//${split[0]}_$i.${split[1]}
done

test_body() {
    # test code
    # First running simple code
    python video_label_example.py

    nosetests --verbose ../../../dali/test/python/test_video_pipeline.py
}

source ../../../qa/test_template.sh

popd
