if (BUILD_BENCHMARK)
  # Get all the benchmark srcs
  file(GLOB tmp *_bench.cc)
  set(NDLL_BENCHMARK_SRCS ${NDLL_BENCHMARK_SRCS} ${tmp} PARENT_SCOPE)
endif()
