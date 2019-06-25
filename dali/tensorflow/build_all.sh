#!/bin/bash

set -o xtrace

TF_VERSIONS=${1:-"1.14.0"}
DOCKER_IMAGE_PREFIX=${2:-"gitlab-master.nvidia.com:5005/dl/dali/dali-gh-mirror:dali_tf_"}

if ! test -f whl/nvidia_dali*.whl; then
    echo "DALI wheel should be present in ./whl directory"
    exit 1
fi

for TF_VERSION in ${TF_VERSIONS}; do
    TF_VERSION_UNDERSCORE=$(echo $TF_VERSION | sed 's/\([0-9]\+\)\.\([0-9]\+\).*/\1_\2/')
    DOCKER_IMAGE="${DOCKER_IMAGE_PREFIX}${TF_VERSION_UNDERSCORE}_builder"
    docker pull ${DOCKER_IMAGE} || docker build -t ${DOCKER_IMAGE} --build-arg "TF_VERSION=${TF_VERSION}" . && docker push ${DOCKER_IMAGE}
    ls -l
    docker run --rm -v `pwd`:/opt/dali -w /opt/dali ${DOCKER_IMAGE} /bin/bash -exc "ls -l /opt/dali && ls -l && source build_in_custom_op_docker.sh"
done
