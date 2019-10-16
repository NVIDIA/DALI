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

# Add all subdirectories
add_subdirectory(data)
add_subdirectory(executor)
add_subdirectory(graph)
add_subdirectory(operator)
add_subdirectory(proto)
add_subdirectory(util)
add_subdirectory(workspace)

# Get all the source files and dump test files
collect_headers(DALI_INST_HDRS PARENT_SCOPE)
collect_sources(DALI_SRCS PARENT_SCOPE)
collect_test_sources(DALI_TEST_SRCS PARENT_SCOPE)

if (BUILD_BENCHMARK)
  # Get all the benchmark srcs
  file(GLOB tmp *_bench.cc)
  set(DALI_BENCHMARK_SRCS ${DALI_BENCHMARK_SRCS} ${tmp} PARENT_SCOPE)
endif()

# Get the protobuf files for pipe serialization
protobuf_generate_cpp(DALI_PROTO_SRCS DALI_PROTO_HEADERS proto/dali.proto)
add_library(DALI_PROTO OBJECT ${DALI_PROTO_HEADERS} ${DALI_PROTO_SRCS})
install(FILES ${DALI_PROTO_HEADERS} DESTINATION include/dali/pipeline)
