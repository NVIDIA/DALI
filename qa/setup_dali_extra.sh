#!/bin/bash

# Fetch test data
export DALI_EXTRA_PATH=${DALI_EXTRA_PATH:-/opt/dali_extra}

#export DALI_EXTRA_URL=${DALI_EXTRA_URL:-"https://github.com/NVIDIA/DALI_extra.git"}

DALI_EXTRA_URL="https://github.com/jantonguirao/DALI_extra/"

DIR=$(cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null && pwd )
DALI_EXTRA_VERSION_PATH="${DIR}/../DALI_EXTRA_VERSION"
read -r DALI_EXTRA_VERSION < ${DALI_EXTRA_VERSION_PATH}
echo "Using DALI_EXTRA_VERSION = ${DALI_EXTRA_VERSION}"
if [ ! -d "$DALI_EXTRA_PATH" ] ; then
    git clone "$DALI_EXTRA_URL" "$DALI_EXTRA_PATH"
    git checkout split_db_images_into_imagenet_classes
fi

pushd "$DALI_EXTRA_PATH"
git checkout split_db_images_into_imagenet_classes
popd
