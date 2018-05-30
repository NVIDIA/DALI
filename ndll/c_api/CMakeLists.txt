# Get all the source files
file(GLOB tmp *.cc)
set(NDLL_SRCS ${NDLL_SRCS} ${tmp} PARENT_SCOPE)
