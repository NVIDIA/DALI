#!/bin/bash

# Fetch test data
export DALI_EXTRA_PATH=/opt/dali_extra
DALI_EXTRA_URL="https://github.com/NVIDIA/DALI_extra.git"
if [ ! -d "$DALI_EXTRA_PATH" ] ; then
    git clone --depth=1 "$DALI_EXTRA_URL" "$DALI_EXTRA_PATH"
else 
    pushd "$DALI_EXTRA_PATH"
    git fetch origin master
    git checkout origin/master
    popd
fi
