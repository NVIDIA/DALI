# Folder is header only save for tests
if (BUILD_TEST)
  file(GLOB tmp *_test.cc)
  set(NDLL_TEST_SRCS ${NDLL_TEST_SRCS} ${tmp} PARENT_SCOPE)
endif()
