# Folder is header only save for tests
if (BUILD_TEST)
  file(GLOB tmp *_test.cc)
  set(NDLL_TEST_SRCS ${NDLL_TEST_SRCS} ${tmp} PARENT_SCOPE)
endif()

if (USE_PROTOBUF)
  # caffe, C2
  protobuf_generate_cpp(CAFFE_PROTO_SRCS CAFFE_PROTO_HEADERS proto/caffe.proto)
  protobuf_generate_cpp(CAFFE2_PROTO_SRCS CAFFE2_PROTO_HEADERS proto/caffe2.proto)

  # TF
  protobuf_generate_cpp(TF_PROTO_SRCS TF_PROTO_HEADERS proto/example.proto proto/feature.proto)

  add_library(CAFFE_PROTO OBJECT ${CAFFE_PROTO_HEADERS} ${CAFFE_PROTO_SRCS})
  add_library(CAFFE2_PROTO OBJECT ${CAFFE2_PROTO_HEADERS} ${CAFFE2_PROTO_SRCS})
  add_library(TF_PROTO OBJECT ${TF_PROTO_HEADERS} ${TF_PROTO_SRCS})
endif()
