# Get all the source files
file(GLOB tmp *.cc *.cu)
set(DALI_SRCS ${DALI_SRCS} ${tmp} PARENT_SCOPE)
