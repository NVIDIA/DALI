cmake_minimum_required(VERSION 3.25.2)
set(CMAKE_CUDA_ARCHITECTURES "80;90")

project(naive_histogram_plugin LANGUAGES CUDA CXX C)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_C_STANDARD 11)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

include_directories(SYSTEM "${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}")

execute_process(
        COMMAND python -c "import nvidia.dali as dali; print(dali.sysconfig.get_lib_dir())"
        OUTPUT_VARIABLE DALI_LIB_DIR)
string(STRIP ${DALI_LIB_DIR} DALI_LIB_DIR)

execute_process(
        COMMAND python -c "import nvidia.dali as dali; print(\" \".join(dali.sysconfig.get_compile_flags()))"
        OUTPUT_VARIABLE DALI_COMPILE_FLAGS)
string(STRIP ${DALI_COMPILE_FLAGS} DALI_COMPILE_FLAGS)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${DALI_COMPILE_FLAGS} ")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} ${DALI_COMPILE_FLAGS} ")
link_directories("${DALI_LIB_DIR}")

set(PLUGIN_SOURCES
        ${CMAKE_CURRENT_SOURCE_DIR}/naive_histogram.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/naive_histogram.cu
        )

add_library(naivehistogram SHARED ${PLUGIN_SOURCES})
target_link_libraries(naivehistogram dali)
