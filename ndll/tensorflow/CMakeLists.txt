if (BUILD_TENSORFLOW)

  file(GLOB tmp *.cc)
  set(NDLL_TF_SRCS ${NDLL_TF_SRCS} ${tmp} PARENT_SCOPE)

endif()
