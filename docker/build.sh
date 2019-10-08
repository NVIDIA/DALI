#!/bin/bash -xe

usage="ENV1=VAL1 ENV2=VAL2 [...] $(basename "$0") [-h] -- this is simple, one click build script for DALI utilizing Docker as
a build environment

To change build configuration please export appropriate env variables (for exact meaning please check the README):
PYVER=[default 3.7]
CUDA_VERSION=[default 10, accepts also 9]
NVIDIA_BUILD_ID=[default 12345]
CREATE_WHL=[default YES]
CREATE_RUNNER=[default NO]
BUILD_TF_PLUGIN=[default NO]
DALI_BUILD_FLAVOR=[default is empty]
CMAKE_BUILD_TYPE=[default is Release]
BUILD_INHOST=[create build dir with object outside docker, just mount it as a volume, default is YES]
REBUILD_BUILDERS=[default is NO]
REBUILD_MANYLINUX=[default is NO]
DALI_BUILD_DIR=[default is build-docker-\${CMAKE_BUILD_TYPE}-\${PYV}-\${CUDA_VERSION}]

where:
    -h  show this help text"

while getopts 'h' option; do
  case "$option" in
    h) echo "$usage"
       exit
       ;;
   \?) printf "illegal option: -%s\n" "$OPTARG" >&2
       echo "$usage" >&2
       exit 1
       ;;
  esac
done
shift $((OPTIND - 1))

#########Set Me###############
export PYVER=${PYVER:-3.5}
export PYV=${PYVER/./}
export CUDA_VERSION=${CUDA_VERSION:-10}
export NVIDIA_BUILD_ID=${NVIDIA_BUILD_ID:-12345}
export CREATE_WHL=${CREATE_WHL:-YES}
export CREATE_RUNNER=${CREATE_RUNNER:-NO}
export DALI_BUILD_FLAVOR=${DALI_BUILD_FLAVOR}
export CMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
export BUILD_INHOST=${BUILD_INHOST:-YES}
export REBUILD_BUILDERS=${REBUILD_BUILDERS:-NO}
export REBUILD_MANYLINUX=${REBUILD_MANYLINUX:-NO}
export BUILD_TF_PLUGIN=${BUILD_TF_PLUGIN:-NO}
export DALI_BUILD_DIR=${DALI_BUILD_DIR:-build-docker-${CMAKE_BUILD_TYPE}-${PYV}-${CUDA_VERSION}}
#################################
export DEPS_IMAGE=nvidia/dali:cu${CUDA_VERSION}.deps
export CUDA_DEPS_IMAGE=nvidia/dali:cuda${CUDA_VERSION}.toolkit
export BUILDER=nvidia/dali:py${PYV}_cu${CUDA_VERSION}.build
export BUILDER_WHL=nvidia/dali:py${PYV}_cu${CUDA_VERSION}.build_whl
export BUILDER_DALI_TF_BASE_MANYLINUX1=nvidia/dali:py${PYV}_cu${CUDA_VERSION}.build_tf_base_manylinux1
export BUILDER_DALI_TF_BASE_MANYLINUX2010=nvidia/dali:py${PYV}_cu${CUDA_VERSION}.build_tf_base_manylinux2010
export BUILDER_DALI_TF_BASE_WITH_WHEEL=nvidia/dali:py${PYV}_cu${CUDA_VERSION}.build_tf_base_with_whl
export BUILDER_DALI_TF_MANYLINUX1=nvidia/dali:py${PYV}_cu${CUDA_VERSION}.build_tf_manylinux1
export BUILDER_DALI_TF_MANYLINUX2010=nvidia/dali:py${PYV}_cu${CUDA_VERSION}.build_tf_manylinux2010
export BUILDER_DALI_TF_SDIST=nvidia/dali:py${PYV}_cu${CUDA_VERSION}.build_tf_sdist
export RUN_IMG=nvidia/dali:py${PYV}_cu${CUDA_VERSION}.run
export GIT_SHA=$(git rev-parse HEAD)
export DALI_TIMESTAMP=$(date +%Y%m%d)

set -o errexit

if [ $CUDA_VERSION != "9" ] && [ $CUDA_VERSION != "10" ]
then
    echo "Wrong CUDA_VERSION=$CUDA_VERSION provided. Only `9` and `10` are supported"
    exit 1
fi

# build manylinux3 if needed
if [[ "$(docker images -q manylinux3_x86_64 2> /dev/null)" == "" || "$REBUILD_MANYLINUX" != "NO" ]]; then
    pushd ../third_party/manylinux/
    git checkout 96b47a25673b33c728e49099a3a6b1bf503a18c2 || echo -e "Did you forget to \`git clone --recursive\`? Try this:\n" \
                                                                    "  git submodule sync --recursive && \n" \
                                                                    "  git submodule update --init --recursive && \n"
    git am ../../docker/0001-An-approximate-manylinux3.patch
    PLATFORM=$(uname -m) TRAVIS_COMMIT=latest ./build.sh
    popd
fi

pushd ../
# build deps image if needed
if [[ "$(docker images -q ${DEPS_IMAGE} 2> /dev/null)" == "" || "$(docker images -q ${CUDA_DEPS_IMAGE} 2> /dev/null)" == "" || "$REBUILD_BUILDERS" != "NO" ]]; then
    echo "Build deps: " ${DEPS_IMAGE}
    docker build -t ${CUDA_DEPS_IMAGE} -f docker/Dockerfile.cuda${CUDA_VERSION}.deps .
    docker build -t ${DEPS_IMAGE} --build-arg "FROM_IMAGE_NAME"=manylinux3_x86_64 --build-arg "CUDA_IMAGE=${CUDA_DEPS_IMAGE}" -f docker/Dockerfile.deps .
fi

# build builder image if needed
if [[ "$(docker images -q ${BUILDER} 2> /dev/null)" == "" || "$(docker images -q ${BUILDER_DALI_TF_BASE_MANYLINUX1} 2> /dev/null)" == "" || "$(docker images -q ${BUILDER_DALI_TF_BASE_MANYLINUX2010} 2> /dev/null)" || "$REBUILD_BUILDERS" != "NO" ]]; then
    echo "Build light image:" ${BUILDER}
    docker build -t ${BUILDER} --build-arg "DEPS_IMAGE_NAME=${DEPS_IMAGE}" --build-arg "PYVER=${PYVER}" --build-arg "PYV=${PYV}" --build-arg "NVIDIA_BUILD_ID=${NVIDIA_BUILD_ID}" \
                               --build-arg "NVIDIA_DALI_BUILD_FLAVOR=${DALI_BUILD_FLAVOR}" --build-arg "GIT_SHA=${GIT_SHA}" --build-arg "DALI_TIMESTAMP=${DALI_TIMESTAMP}" \
                               --build-arg "CMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}" --target builder -f docker/Dockerfile .

    echo "Build DALI TF base (manylinux1)"
    TF_CUSTOM_OP_IMAGE_MANYLINUX1="tensorflow/tensorflow:custom-op-gpu-ubuntu14"
    docker build -t ${BUILDER_DALI_TF_BASE_MANYLINUX1} \
           --build-arg "TF_CUSTOM_OP_IMAGE=${TF_CUSTOM_OP_IMAGE_MANYLINUX1}" \
           --build-arg "CUDA_IMAGE=${CUDA_DEPS_IMAGE}" \
           --build-arg "PYVER=${PYVER}" --build-arg "PYV=${PYV}" \
           -f docker/Dockerfile.customopbuilder.clean .
    echo "Build DALI TF base (manylinux2010)"
    TF_CUSTOM_OP_IMAGE_MANYLINUX2010="tensorflow/tensorflow:custom-op-gpu-ubuntu16"
    docker build -t ${BUILDER_DALI_TF_BASE_MANYLINUX2010} \
           --build-arg "TF_CUSTOM_OP_IMAGE=${TF_CUSTOM_OP_IMAGE_MANYLINUX2010}" \
           --build-arg "CUDA_IMAGE=${CUDA_DEPS_IMAGE}" \
           --build-arg "PYVER=${PYVER}" --build-arg "PYV=${PYV}" \
           -f docker/Dockerfile.customopbuilder.clean .

fi

if [ "$BUILD_INHOST" = "YES" ]; then
    # build inside the source tree
    docker run --rm -u 1000:1000 -v $(pwd):/opt/dali ${BUILDER} /bin/bash -c "mkdir -p /opt/dali/${DALI_BUILD_DIR} && \
                                        cd /opt/dali/${DALI_BUILD_DIR} &&         \
                                        rm -rf /opt/dali/${DALI_BUILD_DIR}/nvidia* && \
                                        CMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}      \
                                        BUILD_TEST=${BUILD_TEST}                  \
                                        BUILD_BENCHMARK=${BUILD_BENCHMARK}        \
                                        BUILD_NVTX=${BUILD_NVTX}        \
                                        BUILD_PYTHON=${BUILD_PYTHON}              \
                                        BUILD_LMDB=${BUILD_LMDB}                  \
                                        BUILD_JPEG_TURBO=${BUILD_JPEG_TURBO}      \
                                        BUILD_NVJPEG=${BUILD_NVJPEG}              \
                                        BUILD_LIBTIFF=${BUILD_LIBTIFF}            \
                                        BUILD_NVOF=${BUILD_NVOF}                  \
                                        BUILD_NVDEC=${BUILD_NVDEC}                \
                                        BUILD_NVML=${BUILD_NVML}                  \
                                        VERBOSE_LOGS=${VERBOSE_LOGS}              \
                                        WERROR=${WERROR}                          \
                                        BUILD_WITH_ASAN=${BUILD_WITH_ASAN}        \
                                        NVIDIA_BUILD_ID=${NVIDIA_BUILD_ID}        \
                                        GIT_SHA=${GIT_SHA}                        \
                                        DALI_TIMESTAMP=${DALI_TIMESTAMP}          \
                                        NVIDIA_DALI_BUILD_FLAVOR=${DALI_BUILD_FLAVOR} \
                                        /opt/dali/docker/build_helper.sh &&       \
                                        rm -rf /opt/dali/${DALI_BUILD_DIR}/nvidia* && \
                                        cp /wheelhouse/* ./"
else
    echo "Build image:" ${BUILDER_WHL}
    docker build -t ${BUILDER_WHL} --build-arg "DEPS_IMAGE_NAME=${DEPS_IMAGE}" --build-arg "PYVER=${PYVER}" --build-arg "PYV=${PYV}" --build-arg "NVIDIA_BUILD_ID=${NVIDIA_BUILD_ID}" \
                                   --build-arg "NVIDIA_DALI_BUILD_FLAVOR=${DALI_BUILD_FLAVOR}" --build-arg "GIT_SHA=${GIT_SHA}" --build-arg "DALI_TIMESTAMP=${DALI_TIMESTAMP}" \
                                   --build-arg "CMAKE_BUILD_TYPE=${BUILD_TYPE}" --cache-from "${BUILDER}" .
fi


if [ "$CREATE_RUNNER" = "YES" ]; then
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
    export BUILDER_TMP=${BUILDER_WHL}
    # for intree build we don't have docker image with whl inside so create one
    if [ "$BUILD_INHOST" = "YES" ]; then
        export BUILDER_TMP=${BUILDER}_tmp
        DOCKER_FILE_TMP=$(mktemp)
        echo -e "FROM scratch\n"          \
                "COPY ./${DALI_BUILD_DIR}/nvidia* /wheelhouse/\n" > ${DOCKER_FILE_TMP}
        docker build -t ${BUILDER_TMP} -f ${DOCKER_FILE_TMP} .
        rm ${DOCKER_FILE_TMP}
    fi
    docker build -t ${RUN_IMG} --build-arg "BUILD_IMAGE_NAME=${BUILDER_TMP}" --build-arg "CUDA_IMAGE_NAME=${CUDA_IMAGE_NAME}" --build-arg "PYVER=${PYVER}" --build-arg "PYV=${PYV}" -f Docker_run_cuda .
    # remove scratch image
    if [ "$BUILD_INHOST" = "YES" ]; then
        docker rmi ${BUILDER_TMP}
    fi
fi

mkdir -p ./wheelhouse/

if [ "$CREATE_WHL" = "YES" ]; then
    if [ "$BUILD_INHOST" = "YES" ]; then
        cp $(pwd)/${DALI_BUILD_DIR}/nvidia* ./wheelhouse/
    else
        export CONTAINER="extract-tmp"
        docker create --name "${CONTAINER}" ${BUILDER_WHL}
        docker cp "${CONTAINER}:/wheelhouse/." "./wheelhouse/"
        docker rm -f "${CONTAINER}"
    fi
fi


if [ "$BUILD_TF_PLUGIN" = "YES" ]; then

    mkdir -p dali_tf_plugin/whl
    cp ./wheelhouse/*.whl dali_tf_plugin/whl/

# TODO: Enable when we figure out how to do pip install without root in build_in_custom_op_docker.sh

# if [ "$BUILD_INHOST" = "YES" ]; then
#     docker build -t ${BUILDER_DALI_TF_BASE_WITH_WHEEL} \
#            --build-arg "TF_CUSTOM_OP_BUILDER_IMAGE=${BUILDER_DALI_TF_BASE}" \
#            -f docker/Dockerfile_dali_tf \
#            --target base_with_wheel \
#            .
#     nvidia-docker run --name ${DALI_TF_BUILDER_CONTAINER} \
#            --user root -v $(pwd):/opt/dali -v ${tmp_wheelhouse}:/dali_tf_sdist \
#            ${BUILDER_DALI_TF_BASE_WITH_WHEEL} /bin/bash -c \
#            "cd /opt/dali/dali_tf_plugin &&                \
#             NVIDIA_BUILD_ID=${NVIDIA_BUILD_ID}            \
#             GIT_SHA=${GIT_SHA}                            \
#             DALI_TIMESTAMP=${DALI_TIMESTAMP}              \
#             NVIDIA_DALI_BUILD_FLAVOR=${DALI_BUILD_FLAVOR} \
#             /bin/bash build_in_custom_op_docker.sh"
    # else

    echo "Build image:" ${BUILDER_DALI_TF_MANYLINUX1}
    docker build -t ${BUILDER_DALI_TF_MANYLINUX1} -f docker/Dockerfile_dali_tf \
        --build-arg "TF_CUSTOM_OP_BUILDER_IMAGE=${BUILDER_DALI_TF_BASE_MANYLINUX1}" \
        .
    export DALI_TF_BUILDER_CONTAINER_MANYLINUX1="${BUILDER_DALI_TF}_container_manylinux1"
    nvidia-docker run --name ${DALI_TF_BUILDER_CONTAINER_MANYLINUX1} ${BUILDER_DALI_TF_MANYLINUX1} /bin/bash -c 'source /opt/dali/dali_tf_plugin/build_in_custom_op_docker.sh'
    docker cp "${DALI_TF_BUILDER_CONTAINER_MANYLINUX1}:/prebuilt/." "prebuilt_manylinux1"
    docker rm -f "${DALI_TF_BUILDER_CONTAINER_MANYLINUX1}"

    echo "Build image:" ${BUILDER_DALI_TF_MANYLINUX2010}
    docker build -t ${BUILDER_DALI_TF_MANYLINUX2010} -f docker/Dockerfile_dali_tf \
        --build-arg "TF_CUSTOM_OP_BUILDER_IMAGE=${BUILDER_DALI_TF_BASE_MANYLINUX2010}" \
        .
    export DALI_TF_BUILDER_CONTAINER_MANYLINUX2010="${BUILDER_DALI_TF}_container_manylinux2010"
    nvidia-docker run --name ${DALI_TF_BUILDER_CONTAINER_MANYLINUX2010} ${BUILDER_DALI_TF_MANYLINUX2010} /bin/bash -c 'source /opt/dali/dali_tf_plugin/build_in_custom_op_docker.sh'
    docker cp "${DALI_TF_BUILDER_CONTAINER_MANYLINUX2010}:/prebuilt/." "prebuilt_manylinux2010"
    docker rm -f "${DALI_TF_BUILDER_CONTAINER_MANYLINUX2010}"

    mkdir -p ./dali_tf_plugin/prebuilt/;
    cp -r ./prebuilt_manylinux1/* ./prebuilt_manylinux2010/* ./dali_tf_plugin/prebuilt/;
    rm -rf ./prebuilt_manylinux2010/ ./prebuilt_manylinux1/

    docker build -t ${BUILD_DALI_TF_IMAGE_NAME_SDIST} \
           -f docker/Dockerfile_dali_tf \
           --build-arg "TF_CUSTOM_OP_BUILDER_IMAGE=${USED_TF_CUSTOM_OP_BUILDER_CLEAN_IMAGE_NAME_MANYLINUX2010}" \
           --build-arg "NVIDIA_BUILD_ID=${CI_PIPELINE_ID}" \
           --build-arg "NVIDIA_DALI_BUILD_FLAVOR=${DALI_BUILD_FLAVOR}" \
           --build-arg "GIT_SHA=${GIT_SHA}" \
           --build-arg "DALI_TIMESTAMP=${DALI_TIMESTAMP}" \
           . ;
    export DALI_TF_BUILDER_CONTAINER_SDIST="${BUILDER_DALI_TF}_container_sdist"
    nvidia-docker run --name ${DALI_TF_BUILDER_CONTAINER_SDIST} ${BUILDER_DALI_TF_SDIST} /bin/bash -c 'cd /opt/dali/dali_tf_plugin && source make_dali_tf_sdist.sh'
    docker cp "${DALI_TF_BUILDER_CONTAINER_SDIST}:/dali_tf_sdist/." "dali_tf_sdist"
    cp dali_tf_sdist/*.tar.gz wheelhouse/
    docker rm -f "${DALI_TF_BUILDER_CONTAINER_SDIST}"

    rm -rf dali_tf_plugin/whl
    rm -rf dali_tf_sdist/
# fi
fi

popd
