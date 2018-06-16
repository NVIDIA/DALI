# Get all the source files
file(GLOB tmp *.cc *.cu)
set(DALI_SRCS ${DALI_SRCS} ${tmp})
file(GLOB tmp *_test.cc)
remove(DALI_SRCS "${DALI_SRCS}" ${tmp})
set(DALI_SRCS ${DALI_SRCS} PARENT_SCOPE)

if (BUILD_TEST)
  # get all the test srcs
  file(GLOB tmp *_test.cc)
  set(DALI_TEST_SRCS ${DALI_TEST_SRCS} ${tmp} PARENT_SCOPE)
endif()
