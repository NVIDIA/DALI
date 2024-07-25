# Copyright (c) 2019, 2021, 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

add_subdirectory(fused)

collect_headers(DALI_INST_HDRS PARENT_SCOPE)
collect_sources(DALI_OPERATOR_SRCS PARENT_SCOPE)
collect_test_sources(DALI_OPERATOR_TEST_SRCS PARENT_SCOPE)

list(FILTER DALI_OPERATOR_SRCS EXCLUDE REGEX ".*nvjpeg_wrap.cc")
set(DALI_OPERATOR_SRCS ${DALI_OPERATOR_SRCS} PARENT_SCOPE)
