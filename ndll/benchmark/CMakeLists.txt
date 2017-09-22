if (BUILD_BENCHMARK)
  # Get all the benchmark srcs
  file(GLOB tmp *_bench.cc)
  set(NDLL_BENCHMARK_SRCS ${NDLL_BENCHMARK_SRCS} ${tmp} PARENT_SCOPE)
endif()

# DEBUG
set(test_bin "no_gbench")
cuda_add_executable(${test_bin} "test.cc")
add_dependencies(${test_bin} ndll)
target_link_libraries(${test_bin} ${NDLL_LIBS} ndll)

set(test_bin_2 "no_pipeline")
cuda_add_executable(${test_bin_2} "test_2.cc")
add_dependencies(${test_bin_2} ndll)
target_link_libraries(${test_bin_2} ${NDLL_LIBS} ndll)
