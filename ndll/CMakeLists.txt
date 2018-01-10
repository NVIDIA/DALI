# Set variables used by subdirectories
set(NDLL_SRCS)
set(NDLL_TEST_SRCS)
set(NDLL_BENCHMARK_SRCS)

# Build the library
add_subdirectory(image)
add_subdirectory(pipeline)
add_subdirectory(util)

# Collect files in this directory
file(GLOB tmp *.cc)
set(NDLL_SRCS ${NDLL_SRCS} ${tmp})

set(ndll_lib "ndll")

if (BUILD_PROTOBUF)
  cuda_add_library(${ndll_lib} SHARED ${NDLL_SRCS} $<TARGET_OBJECTS:CAFFE_PROTO> $<TARGET_OBJECTS:CAFFE2_PROTO> $<TARGET_OBJECTS:TF_PROTO>)
else()
  cuda_add_library(${ndll_lib} SHARED ${NDLL_SRCS})
endif()
target_link_libraries(${ndll_lib} ${NDLL_LIBS} hybrid_decode)

# Add install rules
install(TARGETS ${ndll_lib} DESTINATION lib)
install(DIRECTORY ${CMAKE_CURRENT_LIST_DIR} DESTINATION
  include FILES_MATCHING PATTERN "*.h" PATTERN "*_test.h" EXCLUDE)

# Build test suite
if (BUILD_TEST)
  # get all test srcs
  add_subdirectory(test)

  set(test_main_bin "run_tests")
  cuda_add_executable(${test_main_bin} "${NDLL_TEST_SRCS}")

  # Link to the ndll lib
  add_dependencies(${test_main_bin} ${ndll_lib})

  # We'll have to add dependency libs
  target_link_libraries(${test_main_bin} ${NDLL_LIBS} ${ndll_lib} gtest)
endif()

# Build benchmark suite
if (BUILD_BENCHMARK)
  # get benchmark main
  add_subdirectory(benchmark)

  set(benchmark_bin "run_benchmarks")
  cuda_add_executable(${benchmark_bin} "${NDLL_BENCHMARK_SRCS}")

  # Link to the ndll lib
  add_dependencies(${benchmark_bin} ${ndll_lib})

  target_link_libraries(${benchmark_bin} ${NDLL_LIBS} ${ndll_lib} benchmark pthread)
endif()

# Build the python bindings
if (BUILD_PYTHON)
  # Get all python srcs
  add_subdirectory(python)
endif()
