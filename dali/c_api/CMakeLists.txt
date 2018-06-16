# Get all the source files
file(GLOB tmp *.cc)
set(DALI_SRCS ${DALI_SRCS} ${tmp} PARENT_SCOPE)
