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


file(GLOB tmp *.cc *.cu)
set(DALI_SRCS ${DALI_SRCS} ${tmp})
file(GLOB tmp *_test.cc)
remove(DALI_SRCS "${DALI_SRCS}" ${tmp})

set(DALI_SRCS ${DALI_SRCS} PARENT_SCOPE)

if (BUILD_TEST)
  file(GLOB tmp *_test.cc)
  set(DALI_TEST_SRCS ${DALI_TEST_SRCS} ${tmp} PARENT_SCOPE)
endif()

# caffe, C2
protobuf_generate_cpp(CAFFE_PROTO_SRCS CAFFE_PROTO_HEADERS proto/caffe.proto)
protobuf_generate_cpp(CAFFE2_PROTO_SRCS CAFFE2_PROTO_HEADERS proto/caffe2.proto)

add_library(CAFFE_PROTO OBJECT ${CAFFE_PROTO_HEADERS} ${CAFFE_PROTO_SRCS})
add_library(CAFFE2_PROTO OBJECT ${CAFFE2_PROTO_HEADERS} ${CAFFE2_PROTO_SRCS})

# TF
if (BUILD_PROTO3)
  protobuf_generate_cpp(TF_PROTO_SRCS TF_PROTO_HEADERS proto/example.proto proto/feature.proto)
  add_library(TF_PROTO OBJECT ${TF_PROTO_HEADERS} ${TF_PROTO_SRCS})
endif()

