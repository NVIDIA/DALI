#!/bin/bash -e
export DALI_USE_EXEC2=0
bash -e ./test_nofw.sh
bash -e ./test_pytorch.sh
