#!/bin/bash

# Fetch test data
export DALI_EXTRA_PATH=${DALI_EXTRA_PATH:-/opt/dali_extra}
export DALI_EXTRA_URL=${DALI_EXTRA_URL:-"https://github.com/NVIDIA/DALI_extra.git"}
export DALI_EXTRA_NO_DOWNLOAD=${DALI_EXTRA_NO_DOWNLOAD}

DIR=$(cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null && pwd )
DALI_EXTRA_VERSION_PATH="${DIR}/../DALI_EXTRA_VERSION"
DALI_EXTRA_VERSION=${DALI_EXTRA_VERSION_SHA:-$(cat ${DALI_EXTRA_VERSION_PATH})}
echo "Using DALI_EXTRA_VERSION = ${DALI_EXTRA_VERSION}"
if [ ! -d "$DALI_EXTRA_PATH" ]; then
    if [ "${DALI_EXTRA_NO_DOWNLOAD}" != "" ]; then
        echo "ERROR: DALI_EXTRA_PATH=${DALI_EXTRA_PATH} does not exist and" \
             "DALI_EXTRA_NO_DOWNLOAD is set. Make sure DALI_extra is mounted in the container." >&2
        exit 1
    fi
    git clone "$DALI_EXTRA_URL" "$DALI_EXTRA_PATH"
fi

pushd "$DALI_EXTRA_PATH"
if [ "${DALI_EXTRA_NO_DOWNLOAD}" == "" ]; then
    git fetch origin ${DALI_EXTRA_VERSION}
fi
git checkout ${DALI_EXTRA_VERSION}
popd
