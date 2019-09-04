#!/bin/bash

# Fetch test data

# User can specify optional argument with target where to download DALI_extra
if [ "$#" -eq 1 ]; then
    export DALI_EXTRA_PATH=$1
else
    export DALI_EXTRA_PATH=/opt/dali_extra
fi

DALI_EXTRA_URL="https://github.com/NVIDIA/DALI_extra.git"
DIR=$(cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null && pwd )
DALI_EXTRA_VERSION_PATH="${DIR}/../DALI_EXTRA_VERSION"
read -r DALI_EXTRA_VERSION < ${DALI_EXTRA_VERSION_PATH}
echo "Using DALI_EXTRA_VERSION = ${DALI_EXTRA_VERSION}"
if [ ! -d "$DALI_EXTRA_PATH" ] ; then
    git clone "$DALI_EXTRA_URL" "$DALI_EXTRA_PATH"
fi

pushd "$DALI_EXTRA_PATH"
git fetch origin ${DALI_EXTRA_VERSION}
git checkout ${DALI_EXTRA_VERSION}
popd
