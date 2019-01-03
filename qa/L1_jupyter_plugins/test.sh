#!/bin/bash -e
# used pip packages
pip_packages="jupyter matplotlib opencv-python mxnet-cu90 tensorflow-gpu torchvision torch"

pushd ../..

# attempt to run jupyter on all example notebooks
mkdir -p idx_files

# We need cmake to run the custom plugin notebook
apt-get update
apt-get install -y --no-install-recommends cmake

cd docs/examples


# Prepare videos for VideoReader example
mkdir -p videos
container_name=prepared.mp4
# Download video sample
wget -q -O ${container_name} https://download.blender.org/durian/trailer/sintel_trailer-720p.mp4
IFS='.' read -a splitted <<< "$container_name"
for i in {0..4};
do
  ffmpeg -ss 00:00:${i}0 -t 00:00:10 -i $container_name -vcodec copy -acodec copy videos/${splitted[0]}_$i.${splitted[1]}
done

test_body() {
    # test code
    find */* -name "*.ipynb" | xargs -i jupyter nbconvert \
                   --to notebook --inplace --execute \
                   --ExecutePreprocessor.kernel_name=python${PYVER:0:1} \
                   --ExecutePreprocessor.timeout=600 {}
    find */* -name "main.py" | xargs -i python${PYVER:0:1} {} -t
}

source ../../qa/test_template.sh

popd
