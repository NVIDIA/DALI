add_subdirectory(loader)
add_subdirectory(parser)

list(APPEND NDLL_SRCS "${CMAKE_CURRENT_SOURCE_DIR}/file_reader_op.cc")

if (BUILD_LMDB)
  list(APPEND NDLL_SRCS "${CMAKE_CURRENT_SOURCE_DIR}/caffe_reader_op.cc")
  list(APPEND NDLL_SRCS "${CMAKE_CURRENT_SOURCE_DIR}/caffe2_reader_op.cc")
endif()

#tfrecord
if (BUILD_PROTO3)
  list(APPEND NDLL_SRCS "${CMAKE_CURRENT_SOURCE_DIR}/tfrecord_reader_op.cc")
endif()

list(APPEND NDLL_SRCS "${CMAKE_CURRENT_SOURCE_DIR}/mxnet_reader_op.cc")

set(NDLL_SRCS ${NDLL_SRCS} PARENT_SCOPE)

if (BUILD_TEST)
  # get all the test srcs
  list(APPEND NDLL_TEST_SRCS "${CMAKE_CURRENT_SOURCE_DIR}/reader_op_test.cc")
  set(NDLL_TEST_SRCS ${NDLL_TEST_SRCS} PARENT_SCOPE)
endif()
