# Get all the source files
file(GLOB tmp *.cc)
set(NDLL_SRCS ${NDLL_SRCS} ${tmp} PARENT_SCOPE)

if (BUILD_TEST)
  # get all the test srcs
  file(GLOB tmp *_test.cc)
  set(NDLL_TEST_SRCS ${NDLL_TEST_SRCS} ${tmp} PARENT_SCOPE)
endif()
