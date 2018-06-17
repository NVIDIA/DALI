# Copyright (c) 2017-2018, NVIDIA CORPORATION. All rights reserved.
if (BUILD_BENCHMARK)
  # Get basic benchmark sources that don't depend on any other options
  set(DALI_BENCHMARK_SRCS
    #"${CMAKE_CURRENT_SOURCE_DIR}/makecontiguous_bench.cc"
    "${CMAKE_CURRENT_SOURCE_DIR}/resnet50_bench.cc"
    "${CMAKE_CURRENT_SOURCE_DIR}/resnet50_nvjpeg_bench.cc"
    "${CMAKE_CURRENT_SOURCE_DIR}/dali_bench.cc"
    "${CMAKE_CURRENT_SOURCE_DIR}/file_reader_alexnet_bench.cc"
    "${CMAKE_CURRENT_SOURCE_DIR}/decoder_bench.cc"
  )

  if (BUILD_LMDB)
    list(APPEND DALI_BENCHMARK_SRCS "${CMAKE_CURRENT_SOURCE_DIR}/caffe_alexnet_bench.cc")
    list(APPEND DALI_BENCHMARK_SRCS "${CMAKE_CURRENT_SOURCE_DIR}/caffe2_alexnet_bench.cc")
  endif()

  set(DALI_BENCHMARK_SRCS ${DALI_BENCHMARK_SRCS} PARENT_SCOPE)
endif()
