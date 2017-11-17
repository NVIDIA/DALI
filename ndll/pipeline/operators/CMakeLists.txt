# Get all the source files and dump test files
file(GLOB tmp *.cc)
set(NDLL_SRCS ${NDLL_SRCS} ${tmp})
file(GLOB tmp *_test.cc)
remove(NDLL_SRCS "${NDLL_SRCS}" ${tmp})

# DEBUG: Don't worry about all the ops until the refactor is reasonably stable
# remove(NDLL_SRCS "${NDLL_SRCS}" "/home/tgale/ndll/ndll/pipeline/operators/copy_op.cc")
remove(NDLL_SRCS "${NDLL_SRCS}" "/home/tgale/ndll/ndll/pipeline/operators/crop_mirror_normalize_permute_op.cc")
remove(NDLL_SRCS "${NDLL_SRCS}" "/home/tgale/ndll/ndll/pipeline/operators/hybrid_jpg_decoder.cc")
remove(NDLL_SRCS "${NDLL_SRCS}" "/home/tgale/ndll/ndll/pipeline/operators/normalize_permute_op.cc")
remove(NDLL_SRCS "${NDLL_SRCS}" "/home/tgale/ndll/ndll/pipeline/operators/resize_crop_mirror_op.cc")
remove(NDLL_SRCS "${NDLL_SRCS}" "/home/tgale/ndll/ndll/pipeline/operators/resize_op.cc")
remove(NDLL_SRCS "${NDLL_SRCS}" "/home/tgale/ndll/ndll/pipeline/operators/tjpg_decoder.cc")

set(NDLL_SRCS ${NDLL_SRCS} PARENT_SCOPE)

if (BUILD_TEST)
  # get all the test srcs
  file(GLOB tmp *_test.cc)

  # DEBUG
  remove(tmp "${tmp}" "/home/tgale/ndll/ndll/pipeline/operators/hybrid_jpg_decoder_test.cc")
  
  set(NDLL_TEST_SRCS ${NDLL_TEST_SRCS} ${tmp} PARENT_SCOPE)
endif()
