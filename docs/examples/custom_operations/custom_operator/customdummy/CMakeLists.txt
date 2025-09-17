cmake_minimum_required(VERSION 3.25.2)
set(CMAKE_CUDA_ARCHITECTURES "80;90")

project(custom_dummy_plugin LANGUAGES CUDA CXX C)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_C_STANDARD 11)

# TODO(klecki): When the test container gets a CMake that supports C++17 as a proper option,
# swap those lines
# set(CMAKE_CUDA_STANDARD 17)
# set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -std=c++17")

include_directories(SYSTEM "${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}")

execute_process(
        COMMAND python -c "import nvidia.dali as dali; \
                           print(dali.sysconfig.get_lib_dir())"
        OUTPUT_VARIABLE DALI_LIB_DIR)
string(STRIP ${DALI_LIB_DIR} DALI_LIB_DIR)

execute_process(
        COMMAND python -c "import nvidia.dali as dali; print(\" \
                          \".join(dali.sysconfig.get_compile_flags()))"
        OUTPUT_VARIABLE DALI_COMPILE_FLAGS)
string(STRIP ${DALI_COMPILE_FLAGS} DALI_COMPILE_FLAGS)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${DALI_COMPILE_FLAGS} ")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} ${DALI_COMPILE_FLAGS} ")
link_directories("${DALI_LIB_DIR}")

add_library(dali_customdummy SHARED dummy.cc dummy.cu)
target_link_libraries(dali_customdummy dali)
