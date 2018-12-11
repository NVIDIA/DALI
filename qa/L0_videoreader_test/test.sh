#!/bin/bash -e

pip_packages="numpy"

pushd ../..

cd docs/examples/video

# Download video sample
wget -q -O prepared.mp4 https://download.blender.org/durian/trailer/sintel_trailer-720p.mp4

test_body() {
    # test code
    ls
    python video_example.py
}

source ../../../qa/test_template.sh

popd
