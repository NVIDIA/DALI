if (BUILD_BENCHMARK)
  # Get all the benchmark srcs
  file(GLOB tmp *_bench.cc)

  # DEBUG
  remove(tmp "${tmp}" "/home/tgale/ndll/ndll/benchmark/c2_resnet50_pipe_bench.cc")

  set(NDLL_BENCHMARK_SRCS ${NDLL_BENCHMARK_SRCS} ${tmp} PARENT_SCOPE)
endif()
