# Copyright (c) 2017-2018, NVIDIA CORPORATION. All rights reserved.
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

collect_headers(DALI_INST_HDRS PARENT_SCOPE)
collect_sources(DALI_TF_SRCS PARENT_SCOPE)

message( STATUS "Installing sources for dali_tf: ${DALI_TF_SRCS}")
file(COPY        ${DALI_TF_SRCS}
    DESTINATION ${PROJECT_BINARY_DIR}/dali/python/tf_plugin/nvidia/dali/plugin)
