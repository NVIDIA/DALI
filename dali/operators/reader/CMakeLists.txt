# Copyright (c) 2017-2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

add_subdirectory(loader)
add_subdirectory(parser)

collect_headers(DALI_INST_HDRS PARENT_SCOPE) # TODO (ONLY SUPPORTED ONES)

list(APPEND DALI_OPERATOR_SRCS "${CMAKE_CURRENT_SOURCE_DIR}/file_reader_op.cc")
list(APPEND DALI_OPERATOR_SRCS "${CMAKE_CURRENT_SOURCE_DIR}/numpy_reader_op.cc")

if(BUILD_CFITSIO)
  list(APPEND DALI_OPERATOR_SRCS "${CMAKE_CURRENT_SOURCE_DIR}/fits_reader_op.cc")
  list(APPEND DALI_OPERATOR_SRCS "${CMAKE_CURRENT_SOURCE_DIR}/fits_reader_gpu_op.cu")
endif(BUILD_CFITSIO)

if (BUILD_CUFILE)
  list(APPEND DALI_OPERATOR_SRCS "${CMAKE_CURRENT_SOURCE_DIR}/gds_mem.cc")
  list(APPEND DALI_OPERATOR_SRCS "${CMAKE_CURRENT_SOURCE_DIR}/numpy_reader_gpu_op.cc")
  list(APPEND DALI_OPERATOR_SRCS "${CMAKE_CURRENT_SOURCE_DIR}/numpy_reader_gpu_op_impl.cu")
endif()

list(APPEND DALI_OPERATOR_SRCS "${CMAKE_CURRENT_SOURCE_DIR}/sequence_reader_op.cc")

if (BUILD_LIBTAR)
  list(APPEND DALI_OPERATOR_SRCS "${CMAKE_CURRENT_SOURCE_DIR}/webdataset_reader_op.cc")
endif()

if (BUILD_LMDB)
  list(APPEND DALI_OPERATOR_SRCS "${CMAKE_CURRENT_SOURCE_DIR}/caffe_reader_op.cc")
  list(APPEND DALI_OPERATOR_SRCS "${CMAKE_CURRENT_SOURCE_DIR}/caffe2_reader_op.cc")
endif()

#tfrecord
if (BUILD_PROTO3)
  list(APPEND DALI_OPERATOR_SRCS "${CMAKE_CURRENT_SOURCE_DIR}/tfrecord_reader_op.cc")
endif()

list(APPEND DALI_OPERATOR_SRCS "${CMAKE_CURRENT_SOURCE_DIR}/mxnet_reader_op.cc")

list(APPEND DALI_OPERATOR_SRCS "${CMAKE_CURRENT_SOURCE_DIR}/coco_reader_op.cc")

if (BUILD_LIBSND)
   list(APPEND DALI_OPERATOR_SRCS "${CMAKE_CURRENT_SOURCE_DIR}/nemo_asr_reader_op.cc")
endif()

set(DALI_OPERATOR_SRCS ${DALI_OPERATOR_SRCS} PARENT_SCOPE)

if (BUILD_TEST)
  # get all the test srcs
  list(APPEND DALI_OPERATOR_TEST_SRCS "${CMAKE_CURRENT_SOURCE_DIR}/reader_op_test.cc")
  list(APPEND DALI_OPERATOR_TEST_SRCS "${CMAKE_CURRENT_SOURCE_DIR}/coco_reader_op_test.cc")
  if(BUILD_CUFILE)
    list(APPEND DALI_OPERATOR_TEST_SRCS "${CMAKE_CURRENT_SOURCE_DIR}/gds_mem_test.cu")
  endif()
  set(DALI_OPERATOR_TEST_SRCS ${DALI_OPERATOR_TEST_SRCS} PARENT_SCOPE)
endif()
