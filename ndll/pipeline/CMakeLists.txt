# Add all subdirectories
add_subdirectory(data)
add_subdirectory(operators)
add_subdirectory(util)

# Get all the source files and dump test files
file(GLOB tmp *.cc)
set(NDLL_SRCS ${NDLL_SRCS} ${tmp})
file(GLOB tmp *_test.cc)
remove(NDLL_SRCS "${NDLL_SRCS}" ${tmp})
file(GLOB tmp *_bench.cc)
remove(NDLL_SRCS "${NDLL_SRCS}" ${tmp})
set(NDLL_SRCS ${NDLL_SRCS} PARENT_SCOPE)

if (BUILD_TEST)
  # get all the test srcs
  file(GLOB tmp *_test.cc)
  set(NDLL_TEST_SRCS ${NDLL_TEST_SRCS} ${tmp} PARENT_SCOPE)
endif()

if (BUILD_BENCHMARK)
  # Get all the benchmark srcs
  file(GLOB tmp *_bench.cc)
  set(NDLL_BENCHMARK_SRCS ${NDLL_BENCHMARK_SRCS} ${tmp} PARENT_SCOPE)
endif()

# Get the protobuf files for pipe serialization
protobuf_generate_cpp(NDLL_PROTO_SRCS NDLL_PROTO_HEADERS proto/ndll.proto)
add_library(NDLL_PROTO OBJECT ${NDLL_PROTO_HEADERS} ${NDLL_PROTO_SRCS})
