# Build test suite
if (BUILD_TEST)
  # get all test srcs
  message(STATUS "Building test stuff")
  set(NDLL_TEST_SRCS)
  add_subdirectory(test)
  
  set(test_main_bin "run_tests")
  add_executable(${test_main_bin} "${NDLL_TEST_SRCS}")

  # We'll need to add our library as a dependency here
  # add_dependencies()

  # We'll have to add dependency libs
  target_link_libraries(${test_main_bin} gtest)
endif()
