# Set variables used by subdirectories
set(NDLL_SRCS)
set(NDLL_TEST_SRCS)

# Build the library
add_subdirectory(image)

# TODO(tgale): Build to a shared library instead
set(ndll_lib "ndll")
add_library(${ndll_lib} STATIC ${NDLL_SRCS})
target_link_libraries(${ndll_lib} ${NDLL_LIBS})

# Build test suite
if (BUILD_TEST)
  # get all test srcs
  add_subdirectory(test)
  
  set(test_main_bin "run_tests")
  add_executable(${test_main_bin} "${NDLL_TEST_SRCS}")

  # Link to the ndll lib
  add_dependencies(${test_main_bin} ${ndll_lib})

  # We'll have to add dependency libs
  target_link_libraries(${test_main_bin} ${NDLL_LIBS} ${ndll_lib} gtest)
endif()
