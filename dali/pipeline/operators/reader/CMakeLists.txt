add_subdirectory(loader)
add_subdirectory(parser)

list(APPEND DALI_SRCS "${CMAKE_CURRENT_SOURCE_DIR}/file_reader_op.cc")

if (BUILD_LMDB)
  list(APPEND DALI_SRCS "${CMAKE_CURRENT_SOURCE_DIR}/caffe_reader_op.cc")
  list(APPEND DALI_SRCS "${CMAKE_CURRENT_SOURCE_DIR}/caffe2_reader_op.cc")
endif()

#tfrecord
if (BUILD_PROTO3)
  list(APPEND DALI_SRCS "${CMAKE_CURRENT_SOURCE_DIR}/tfrecord_reader_op.cc")
endif()

list(APPEND DALI_SRCS "${CMAKE_CURRENT_SOURCE_DIR}/mxnet_reader_op.cc")

set(DALI_SRCS ${DALI_SRCS} PARENT_SCOPE)

if (BUILD_TEST)
  # get all the test srcs
  list(APPEND DALI_TEST_SRCS "${CMAKE_CURRENT_SOURCE_DIR}/reader_op_test.cc")
  set(DALI_TEST_SRCS ${DALI_TEST_SRCS} PARENT_SCOPE)
endif()
