# Get srcs and build the library
file(GLOB tmp *.cpp *.cu)
set(hybrid_decode_lib "hybrid_decode")
cuda_add_library(${hybrid_decode_lib} STATIC "${tmp}" OPTIONS "-Xcompiler -fPIC")
target_link_libraries(${hybrid_decode_lib} ${HYBRID_DECODE_LIBS})
target_compile_options(${hybrid_decode_lib} PUBLIC -fPIC)

# Add install rule
install(TARGETS ${hybrid_decode_lib} DESTINATION lib)
