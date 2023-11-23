# Copyright (c) 2017-2018, 2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
    "${PROJECT_SOURCE_DIR}/dali/test/dali_test_config.cc"
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
    "${CMAKE_CURRENT_SOURCE_DIR}/color_twist_bench.cc"
    "${CMAKE_CURRENT_SOURCE_DIR}/slice_kernel_bench.cc"
    "${CMAKE_CURRENT_SOURCE_DIR}/slice_kernel_bench.cu"
    "${CMAKE_CURRENT_SOURCE_DIR}/preemphasis_bench.cc"
    "${CMAKE_CURRENT_SOURCE_DIR}/thread_pool_bench.cc"
    "${CMAKE_CURRENT_SOURCE_DIR}/normal_distribution_gpu_bench.cc"
    "${CMAKE_CURRENT_SOURCE_DIR}/file_reader_bench.cc"
    "${CMAKE_CURRENT_SOURCE_DIR}/copy_bench.cc"
    "${CMAKE_CURRENT_SOURCE_DIR}/one_hot_bench.cc"
    "${CMAKE_CURRENT_SOURCE_DIR}/gaussian_blur_bench.cc"
    "${CMAKE_CURRENT_SOURCE_DIR}/flip_bench.cc"
    "${CMAKE_CURRENT_SOURCE_DIR}/cast_bench.cc"
    "${CMAKE_CURRENT_SOURCE_DIR}/coin_flip_bench.cc"
    "${CMAKE_CURRENT_SOURCE_DIR}/transpose_bench.cc"
    "${CMAKE_CURRENT_SOURCE_DIR}/file_reader_fast_forward_bench.cc"
    "${CMAKE_CURRENT_SOURCE_DIR}/checkpointing_bench.cc"
  )

  if (BUILD_LMDB)
    list(APPEND DALI_BENCHMARK_SRCS "${CMAKE_CURRENT_SOURCE_DIR}/caffe_alexnet_bench.cc")
    list(APPEND DALI_BENCHMARK_SRCS "${CMAKE_CURRENT_SOURCE_DIR}/caffe2_alexnet_bench.cc")
  endif()

  adjust_source_file_language_property("${DALI_BENCHMARK_SRCS}")
  add_executable(dali_benchmark "${DALI_BENCHMARK_SRCS}")

  target_link_libraries(dali_benchmark PRIVATE dali dali_operators benchmark ${DALI_LIBS})
  if (BUILD_NVML)
    target_link_libraries(dali_benchmark PRIVATE dynlink_nvml)
  endif(BUILD_NVML)
  target_link_libraries(dali_benchmark PRIVATE "-pie")
  set_target_properties(dali_benchmark PROPERTIES POSITION_INDEPENDENT_CODE ON)
  set_target_properties(dali_benchmark PROPERTIES OUTPUT_NAME "dali_benchmark.bin")

  set_target_properties(dali_benchmark PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/${DALI_WHEEL_DIR}/test")

endif()
