#!/bin/bash -e

pip_packages="numpy"

apt-get update
apt-get install -y wget ffmpeg

pushd ../..

cd docs/examples/video

mkdir -p video_files

container_name=prepared.mp4

# Download video sample
wget -q -O ${container_name} https://download.blender.org/durian/trailer/sintel_trailer-720p.mp4

IFS='.' read -a splitted <<< "$container_name"

for i in {0..4};
do
    ffmpeg -ss 00:00:${i}0 -t 00:00:10 -i $container_name -vcodec copy -acodec copy video_files/${splitted[0]}_$i.${splitted[1]}
done

test_body() {
    # test code
    python video_example.py
}

source ../../../qa/test_template.sh

popd
