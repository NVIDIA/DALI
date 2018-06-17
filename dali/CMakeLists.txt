# Set variables used by subdirectories
set(DALI_SRCS)
set(DALI_TEST_SRCS)
set(DALI_BENCHMARK_SRCS)
set(DALI_TF_SRCS)

# Build the library
add_subdirectory(image)
add_subdirectory(pipeline)
add_subdirectory(util)
add_subdirectory(plugin)
add_subdirectory(c_api)

# Collect files in this directory
file(GLOB tmp *.cc)
set(DALI_SRCS ${DALI_SRCS} ${tmp})

set(dali_lib "dali")

if (BUILD_TENSORFLOW)
  set(customop_lib "dali_tf")
endif()

set(DALI_PROTO_OBJ $<TARGET_OBJECTS:DALI_PROTO>)

if (BUILD_LMDB)
  list(APPEND DALI_PROTO_OBJ $<TARGET_OBJECTS:CAFFE_PROTO> $<TARGET_OBJECTS:CAFFE2_PROTO>)
endif()

if (BUILD_PROTO3)
  list(APPEND DALI_PROTO_OBJ $<TARGET_OBJECTS:TF_PROTO>)
endif()

# cuda_add_library(${dali_lib} SHARED ${DALI_SRCS} $<TARGET_OBJECTS:CAFFE_PROTO> $<TARGET_OBJECTS:CAFFE2_PROTO> $<TARGET_OBJECTS:TF_PROTO> $<TARGET_OBJECTS:DALI_PROTO>)
cuda_add_library(${dali_lib} SHARED ${DALI_SRCS} ${DALI_PROTO_OBJ})
target_link_libraries(${dali_lib} ${DALI_LIBS})

# Add install rules
install(TARGETS ${dali_lib} DESTINATION lib)
install(DIRECTORY ${CMAKE_CURRENT_LIST_DIR} DESTINATION
  include FILES_MATCHING PATTERN "*.h" PATTERN "*_test.h" EXCLUDE)

# Build test suite
if (BUILD_TEST)
  # get all test srcs
  add_subdirectory(test)

  set(test_main_bin "dali_test.bin")
  cuda_add_executable(${test_main_bin} "${DALI_TEST_SRCS}")

  # Link to the dali lib
  add_dependencies(${test_main_bin} ${dali_lib})

  # We'll have to add dependency libs
  target_link_libraries(${test_main_bin} ${DALI_LIBS} ${dali_lib} gtest)

  set_target_properties(${test_main_bin} PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/dali/python/nvidia/dali/test)
endif()

# Build benchmark suite
if (BUILD_BENCHMARK)
  # get benchmark main
  add_subdirectory(benchmark)

  set(benchmark_bin "dali_benchmark.bin")
  cuda_add_executable(${benchmark_bin} "${DALI_BENCHMARK_SRCS}")

  # Link to the dali lib
  add_dependencies(${benchmark_bin} ${dali_lib})

  target_link_libraries(${benchmark_bin} ${DALI_LIBS} ${dali_lib} benchmark pthread)

  set_target_properties(${benchmark_bin} PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/dali/python/nvidia/dali/test)
endif()

# Build the python bindings
if (BUILD_PYTHON)
  # Get all python srcs
  add_subdirectory(python)
endif()

# Build the TensorFlow op
if (BUILD_TENSORFLOW)
  # Get the TF compile et linking flags
  #execute_process(
  #  COMMAND python -c "from __future__ import print_function; import tensorflow as tf; print(\" \".join(tf.sysconfig.get_compile_flags()), end=\"\")"
  #  OUTPUT_VARIABLE TF_CFLAGS)

  #execute_process(
  #  COMMAND python -c "from __future__ import print_function; import tensorflow as tf; print(\" \".join(tf.sysconfig.get_link_flags()), end=\"\")"
  # OUTPUT_VARIABLE TF_LFLAGS)

  execute_process(
    COMMAND python -c "from __future__ import print_function; import sys; paths = sys.path; f = [p for p in paths if \"site\" in p] ; print(f[0], end=\"\")"
    OUTPUT_VARIABLE DIST_PACKAGES_DIR)

  add_subdirectory(tensorflow)

  #set(customop_lib "dali_tf")
  cuda_add_library(${customop_lib} SHARED ${DALI_TF_SRCS})

  # TF <= 1.3
  # target_compile_options(${customop_lib} PRIVATE -D_GLIBCXX_USE_CXX11_ABI=0 -I${TF_INC} -I${TF_INC}/external/nsync/public -lcudart -L /usr/local/cuda/lib64 -L${TF_LIB} -ltensorflow_framework)

  separate_arguments(TF_CFLAGS UNIX_COMMAND "-I${DIST_PACKAGES_DIR}/tensorflow/include -D_GLIBCXX_USE_CXX11_ABI=0")
  target_compile_options(${customop_lib} PRIVATE ${TF_CFLAGS})

  add_dependencies(${customop_lib} ${dali_lib})

  target_link_libraries(${customop_lib} -L${DIST_PACKAGES_DIR}/tensorflow)
  target_link_libraries(${customop_lib} -ltensorflow_framework)
  target_link_libraries(${customop_lib} ${DALI_LIBS} ${dali_lib})

  set_target_properties(${customop_lib} PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/dali/python/nvidia/dali/plugin")
endif()

