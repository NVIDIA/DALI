# Add all subdirectories
add_subdirectory(data)
add_subdirectory(operators)
add_subdirectory(util)
add_subdirectory(workspace)
add_subdirectory(executor)

# Get all the source files and dump test files
file(GLOB tmp *.cc)
set(DALI_SRCS ${DALI_SRCS} ${tmp})
file(GLOB tmp *_test.cc)
remove(DALI_SRCS "${DALI_SRCS}" ${tmp})
file(GLOB tmp *_bench.cc)
remove(DALI_SRCS "${DALI_SRCS}" ${tmp})
set(DALI_SRCS ${DALI_SRCS} PARENT_SCOPE)

if (BUILD_TEST)
  # get all the test srcs
  file(GLOB tmp *_test.cc)
  set(DALI_TEST_SRCS ${DALI_TEST_SRCS} ${tmp} PARENT_SCOPE)
endif()

if (BUILD_BENCHMARK)
  # Get all the benchmark srcs
  file(GLOB tmp *_bench.cc)
  set(DALI_BENCHMARK_SRCS ${DALI_BENCHMARK_SRCS} ${tmp} PARENT_SCOPE)
endif()

# Get the protobuf files for pipe serialization
protobuf_generate_cpp(DALI_PROTO_SRCS DALI_PROTO_HEADERS proto/dali.proto)
add_library(DALI_PROTO OBJECT ${DALI_PROTO_HEADERS} ${DALI_PROTO_SRCS})
install(FILES ${DALI_PROTO_HEADERS} DESTINATION include/dali/pipeline)
