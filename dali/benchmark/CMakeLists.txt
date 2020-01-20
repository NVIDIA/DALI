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

if (BUILD_BENCHMARK)
  collect_headers(DALI_INST_HDRS PARENT_SCOPE)
  # Get basic benchmark sources that don't depend on any other options
  set(DALI_BENCHMARK_SRCS
    #"${CMAKE_CURRENT_SOURCE_DIR}/makecontiguous_bench.cc"
    "${CMAKE_CURRENT_SOURCE_DIR}/resnet50_bench.cc"
    "${CMAKE_CURRENT_SOURCE_DIR}/resnet50_nvjpeg_bench.cc"
    "${CMAKE_CURRENT_SOURCE_DIR}/dali_bench.cc"
    "${CMAKE_CURRENT_SOURCE_DIR}/file_reader_alexnet_bench.cc"
    "${CMAKE_CURRENT_SOURCE_DIR}/decoder_bench.cc"
    "${CMAKE_CURRENT_SOURCE_DIR}/displacement_cpu_bench.cc"
    "${CMAKE_CURRENT_SOURCE_DIR}/crop_bench.cc"
    "${CMAKE_CURRENT_SOURCE_DIR}/crop_mirror_normalize_bench.cc"
    "${CMAKE_CURRENT_SOURCE_DIR}/warp_affine_bench.cc"
    "${CMAKE_CURRENT_SOURCE_DIR}/transpose_cpu_bench.cc"
  )

  if (BUILD_LMDB)
    list(APPEND DALI_BENCHMARK_SRCS "${CMAKE_CURRENT_SOURCE_DIR}/caffe_alexnet_bench.cc")
    list(APPEND DALI_BENCHMARK_SRCS "${CMAKE_CURRENT_SOURCE_DIR}/caffe2_alexnet_bench.cc")
  endif()

  set(DALI_BENCHMARK_SRCS ${DALI_BENCHMARK_SRCS} PARENT_SCOPE)
endif()
