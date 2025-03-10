# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

# This subdirectory contains tests that need non-builtin operators
add_subdirectory(op_test)

# Get all the source files

# The headers here are private and should not be installed.
# collect_headers(DALI_INST_HDRS PARENT_SCOPE)
# The tests from this directory go to dali_test.

collect_sources(DALI_SRCS PARENT_SCOPE)
collect_test_sources(DALI_TEST_SRCS PARENT_SCOPE)
set(DALI_OPERATOR_TEST_SRCS ${DALI_OPERATOR_TEST_SRCS} PARENT_SCOPE)
