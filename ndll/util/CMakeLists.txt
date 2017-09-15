# Get all the source files
file(GLOB tmp *.cc *.cu)
set(NDLL_SRCS ${NDLL_SRCS} ${tmp} PARENT_SCOPE)
