#!/bin/bash -xe
#########Set Me###############
export PYVER=${PYVER:-2.7}
export PYV=${PYVER/./}
export CUDA_VERSION=${CUDA_VERSION:-9}
export NVIDIA_BUILD_ID=${NVIDIA_BUILD_ID:-12345}
export CREATE_WHL=${CREATE_WHL:-YES}
#################################
export DEPS_IMAGE=dali_cu${CUDA_VERSION}.deps
export BUILDER=dali_${PYV}_cu${CUDA_VERSION}.build
export RUN_IMG=dali_${PYV}_cu${CUDA_VERSION}.run

pushd ../
docker build -t ${DEPS_IMAGE} --build-arg "USE_CUDA_VERSION=${CUDA_VERSION}" -f Dockerfile.deps .
echo "Build image:" ${BUILDER}
docker build -t ${BUILDER} --build-arg "DEPS_IMAGE_NAME=${DEPS_IMAGE}" --build-arg "PYVER=${PYVER}" --build-arg "PYV=${PYV}" --build-arg "NVIDIA_BUILD_ID=${NVIDIA_BUILD_ID}" .
echo "Runner image:" ${RUN_IMG}
echo "You can run this image with DALI installed inside, keep in mind to install neccessary FW package as well"
if [ ${CUDA_VERSION} == "9" ] ; then
    export CUDA_IMAGE_NAME="nvidia/cuda:9.0-cudnn7-devel-ubuntu16.04"
elif [ ${CUDA_VERSION} == "10" ] ; then
    export CUDA_IMAGE_NAME="nvidia/cuda:10.0-cudnn7-devel-ubuntu16.04"
else
    echo "**************************************************************"
    echo "Not supported CUDA version"
    echo "**************************************************************"
fi
echo ${CUDA_VERSION}
echo ${CUDA_IMAGE_NAME}

docker build -t ${RUN_IMG} --build-arg "BUILD_IMAGE_NAME=${BUILDER}" --build-arg "CUDA_IMAGE_NAME=${CUDA_IMAGE_NAME}" --build-arg "PYVER=${PYVER}" --build-arg "PYV=${PYV}" -f Docker_run_cuda .

if [ "$CREATE_WHL" = "YES" ]; then
    export CONTAINER="extract-tmp"
    docker create --name "${CONTAINER}" ${BUILDER}
    docker cp "${CONTAINER}:/wheelhouse/" "./"
    docker rm -f "${CONTAINER}"
fi
popd
