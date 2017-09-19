# Get srcs and build the library
file(GLOB tmp *.cpp *.cu)
set(hybrid_decode_lib "hybrid_decode")
cuda_add_library(${hybrid_decode_lib} "${tmp}")
target_link_libraries(${hybrid_decode_lib} ${HYBRID_DECODE_LIBS})
