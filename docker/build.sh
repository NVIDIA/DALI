#!/bin/bash -xe

usage="ENV1=VAL1 ENV2=VAL2 [...] $(basename "$0") [-h] -- this is simple, one click build script for DALI utilizing Docker as
a build environment

To change build configuration please export appropriate env variables (for exact meaning please check the README):
PYVER=[default 3.7]
CUDA_VERSION=[default 10, accepts also 9]
NVIDIA_BUILD_ID=[default 12345]
CREATE_WHL=[default YES]
CREATE_RUNNER=[default NO]
DALI_BUILD_FLAVOR=[default is empty]
CMAKE_BUILD_TYPE=[default is Release]
BUILD_INHOST=[create build dir with object outside docker, just mount it as a volume]
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
export DALI_BUILD_DIR=${DALI_BUILD_DIR:-build-docker-${CMAKE_BUILD_TYPE}-${PYV}-${CUDA_VERSION}}
#################################
export DEPS_IMAGE=dali_cu${CUDA_VERSION}.deps
export BUILDER=dali_${PYV}_cu${CUDA_VERSION}.build
export BUILDER_WHL=dali_${PYV}_cu${CUDA_VERSION}.build_w_whl
export RUN_IMG=dali_${PYV}_cu${CUDA_VERSION}.run
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
if [[ "$(docker images -q ${DEPS_IMAGE} 2> /dev/null)" == "" || "$REBUILD_BUILDERS" != "NO" ]]; then
    echo "Build deps: " ${DEPS_IMAGE}
    docker build -t ${DEPS_IMAGE} --build-arg "FROM_IMAGE_NAME"=manylinux3_x86_64 --build-arg "USE_CUDA_VERSION=${CUDA_VERSION}" -f Dockerfile.deps .
fi

# build builder image if needed
if [[ "$(docker images -q ${BUILDER} 2> /dev/null)" == "" || "$REBUILD_BUILDERS" != "NO" ]]; then
    echo "Build light image:" ${BUILDER}
    docker build -t ${BUILDER} --build-arg "DEPS_IMAGE_NAME=${DEPS_IMAGE}" --build-arg "PYVER=${PYVER}" --build-arg "PYV=${PYV}" --build-arg "NVIDIA_BUILD_ID=${NVIDIA_BUILD_ID}" \
                           --build-arg "NVIDIA_DALI_BUILD_FLAVOR=${DALI_BUILD_FLAVOR}" --build-arg "GIT_SHA=${GIT_SHA}" --build-arg "DALI_TIMESTAMP=${DALI_TIMESTAMP}" \
                           --build-arg "CMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}" --target builder .
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
                                        BUILD_NVOF=${BUILD_NVOF}                  \
                                        BUILD_NVDEC=${BUILD_NVDEC}                \
                                        BUILD_NVML=${BUILD_NVML}                  \
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

tmp_wheelhouse=$(mktemp -d -u)
if [ "$CREATE_WHL" = "YES" ]; then
    if [ "$BUILD_INHOST" = "YES" ]; then
        mkdir -p ${tmp_wheelhouse}
        cp $(pwd)/${DALI_BUILD_DIR}/nvidia* ${tmp_wheelhouse}
    else
        export CONTAINER="extract-tmp"
        docker create --name "${CONTAINER}" ${BUILDER_WHL}
        docker cp "${CONTAINER}:/wheelhouse/" "${tmp_wheelhouse}"
        docker rm -f "${CONTAINER}"
    fi
fi

# Build DALI TF plugin
export CUSTOM_OP_BUILDER_CLEAN_IMAGE_NAME="tf_custom_op_builder_${PYVER}_clean"
docker build -t ${CUSTOM_OP_BUILDER_CLEAN_IMAGE_NAME} --build-arg "PYVER=${PYVER}" --build-arg "PYV=${PYV}" -f docker/Dockerfile.customopbuilder.clean docker

export CUSTOM_OP_BUILDER_IMAGE_NAME="tf_custom_op_builder_${PYVER}_tf_build"
export CUSTOM_OP_BUILDER_CONTAINER="${CUSTOM_OP_BUILDER_IMAGE_NAME}_container"

mkdir -p dali_tf_plugin/whl
cp ${tmp_wheelhouse}/*.whl dali_tf_plugin/whl/
docker build -t ${CUSTOM_OP_BUILDER_IMAGE_NAME} -f docker/Dockerfile_dali_tf \
             --build-arg "TF_CUSTOM_OP_BUILDER_IMAGE=${CUSTOM_OP_BUILDER_CLEAN_IMAGE_NAME}" \
             --build-arg "NVIDIA_BUILD_ID=${NVIDIA_BUILD_ID}" \
             --build-arg "NVIDIA_DALI_BUILD_FLAVOR=${DALI_BUILD_FLAVOR}" \
             --build-arg "GIT_SHA=${GIT_SHA}" \
             --build-arg "DALI_TIMESTAMP=${DALI_TIMESTAMP}" \
             .

nvidia-docker run --name ${CUSTOM_OP_BUILDER_CONTAINER} ${CUSTOM_OP_BUILDER_IMAGE_NAME}
tmp_dali_tf_sdist=$(mktemp -d)
docker cp "${CUSTOM_OP_BUILDER_CONTAINER}:/dali_tf_sdist/" "${tmp_dali_tf_sdist}"
mv ${tmp_dali_tf_sdist}/*.tar.gz ${tmp_wheelhouse}
docker rm -f "${CUSTOM_OP_BUILDER_CONTAINER}"
rm -rf dali_tf_plugin/whl

mkdir -p ./wheelhouse/
mv ${tmp_wheelhouse}/* ./wheelhouse

popd
