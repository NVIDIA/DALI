# Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


#!/bin/bash

DATA_DIR=$1
CHECKPOINT_DIR=$2
RESULT_DIR=$3

docker run -it --rm \
  --gpus='"device=1"' \
  --shm-size=4g \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  -v "$DATA_DIR":/datasets \
  -v "$CHECKPOINT_DIR":/checkpoints/ \
  -v "$RESULT_DIR":/results/ \
  -v $PWD:/code \
  -v $PWD:/workspace/jasper \
  mlperf-rnnt-ref bash
