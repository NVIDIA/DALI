cmake_minimum_required(VERSION 3.12)

project(customdummy CUDA CXX)

execute_process(
        COMMAND python -c "import nvidia.dali as dali; print(dali.sysconfig.get_lib_dir())"
        OUTPUT_VARIABLE DALI_LIB_DIR)
string(STRIP ${DALI_LIB_DIR} DALI_LIB_DIR)

execute_process(
        COMMAND python -c "import nvidia.dali as dali; print(\" \".join(dali.sysconfig.get_compile_flags()))"
        OUTPUT_VARIABLE DALI_COMPILE_FLAGS)
string(STRIP ${DALI_COMPILE_FLAGS} DALI_COMPILE_FLAGS)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${DALI_COMPILE_FLAGS} ")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} ${DALI_COMPILE_FLAGS} ")
link_directories( "${DALI_LIB_DIR}" )
include_directories(${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})

add_library(customdummy SHARED dummy.cc dummy.cu )
target_link_libraries(customdummy dali)
