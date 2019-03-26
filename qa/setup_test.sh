#!/bin/bash

# Fetch test data
export DALI_EXTRA_PATH=/opt/dali_extra
git clone --depth=1 https://github.com/NVIDIA/DALI_extra.git ${DALI_EXTRA_PATH}
