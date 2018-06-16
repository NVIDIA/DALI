if (BUILD_TENSORFLOW)

  file(GLOB tmp *.cc)
  set(DALI_TF_SRCS ${DALI_TF_SRCS} ${tmp} PARENT_SCOPE)

endif()
