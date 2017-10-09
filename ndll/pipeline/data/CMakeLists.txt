# Get all the source files and dump test files
file(GLOB tmp *.cc)
set(NDLL_SRCS ${NDLL_SRCS} ${tmp})
file(GLOB tmp *_test.cc *_test.cu)
remove(NDLL_SRCS "${NDLL_SRCS}" ${tmp})
set(NDLL_SRCS ${NDLL_SRCS} PARENT_SCOPE)

if (BUILD_TEST)
  # get all the test srcs
  file(GLOB tmp *_test.cc *_test.cu)
  set(NDLL_TEST_SRCS ${NDLL_TEST_SRCS} ${tmp} PARENT_SCOPE)
endif()
