#!/bin/bash -xe

usage="ENV1=VAL1 ENV2=VAL2 [...] $(basename "$0") [-h] -- this is simple, one click build script for DALI sdist plugins

To change build configuration please export appropriate env variables:
NVIDIA_DALI_BUILD_FLAVOR=[default is empty]
DALI_PLUGINS_BUILD_DIR=[default is build-docker-plugins]
DALI_PLUGINS_INSTALL_DIR=[default is /tmp/dali_plugins]

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

set -o xtrace
set -e

export NVIDIA_DALI_BUILD_FLAVOR=${NVIDIA_DALI_BUILD_FLAVOR:-}
export NVIDIA_DALI_PLUGINS_INSTALL_DIR=${NVIDIA_DALI_PLUGINS_INSTALL_DIR:-install_plugins}
export GIT_SHA=${GIT_SHA:-$(git rev-parse HEAD)}
export PYTHON_EXECUTABLE=$(which python3 || which python)

mkdir -p ${NVIDIA_DALI_PLUGINS_INSTALL_DIR}

mkdir -p build-plugins-docker
pushd build-plugins-docker

cmake ../plugins \
      -DPYTHON_EXECUTABLE:STRING=${PYTHON_EXECUTABLE} \
      -DCMAKE_INSTALL_PREFIX=${NVIDIA_DALI_PLUGINS_INSTALL_DIR} \
      -DDALI_BUILD_FLAVOR=${NVIDIA_DALI_BUILD_FLAVOR} \
      -DGIT_SHA=${GIT_SHA}
make install

popd