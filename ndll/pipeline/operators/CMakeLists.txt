# Get all the source files and dump test files
file(GLOB tmp *.cc)
set(NDLL_SRCS ${NDLL_SRCS} ${tmp})
file(GLOB tmp *_test.cc)
remove(NDLL_SRCS "${NDLL_SRCS}" ${tmp})

# DEBUG: Don't worry about all the ops until the refactor is reasonably stable
remove(NDLL_SRCS "${NDLL_SRCS}" "/home/tgale/ndll/ndll/pipeline/operators/crop_mirror_normalize_permute_op.cc")

set(NDLL_SRCS ${NDLL_SRCS} PARENT_SCOPE)

if (BUILD_TEST)
  # get all the test srcs
  file(GLOB tmp *_test.cc)
  set(NDLL_TEST_SRCS ${NDLL_TEST_SRCS} ${tmp} PARENT_SCOPE)
endif()
