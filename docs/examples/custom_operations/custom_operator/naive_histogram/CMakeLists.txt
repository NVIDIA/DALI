cmake_minimum_required(VERSION 3.25.2)
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES AND NOT DEFINED ENV{CUDAARCHS})
  set(CMAKE_CUDA_ARCHITECTURES "80;90")
endif()

project(naive_histogram_plugin LANGUAGES CUDA CXX C)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_C_STANDARD 11)

set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

include_directories(SYSTEM "${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}")

# Use the Python interpreter that has DALI installed - the plugin is built against that DALI.
# Prefer the `python` from PATH (the one used to run the plugin) over other interpreters that
# FindPython might find first. Has no effect when `Python_EXECUTABLE` is set, e.g. by pip.
find_program(Python_EXECUTABLE NAMES python python3)
find_package(Python REQUIRED COMPONENTS Interpreter)

execute_process(
        COMMAND ${Python_EXECUTABLE} -c "import nvidia.dali"
        RESULT_VARIABLE DALI_IMPORT_RESULT)
if(NOT DALI_IMPORT_RESULT EQUAL 0)
  message(FATAL_ERROR
          "Could not import nvidia.dali with ${Python_EXECUTABLE}. The plugin is built against the "
          "installed DALI, so DALI must be importable during the build. When installing the plugin "
          "with pip, pass --no-build-isolation.")
endif()

execute_process(
        COMMAND ${Python_EXECUTABLE} -c "import nvidia.dali as dali; print(dali.__version__)"
        OUTPUT_VARIABLE DALI_VERSION
        COMMAND_ERROR_IS_FATAL ANY)
string(STRIP ${DALI_VERSION} DALI_VERSION)

# The CUDA version distinguishes DALI flavors (e.g. CUDA 12 and CUDA 13 builds) of the same release.
execute_process(
        COMMAND ${Python_EXECUTABLE} -c "import nvidia.dali as dali; print(dali.__cuda_version__)"
        OUTPUT_VARIABLE DALI_CUDA_VERSION
        COMMAND_ERROR_IS_FATAL ANY)
string(STRIP ${DALI_CUDA_VERSION} DALI_CUDA_VERSION)

execute_process(
        COMMAND ${Python_EXECUTABLE} -c "import nvidia.dali as dali; print(dali.sysconfig.get_lib_dir())"
        OUTPUT_VARIABLE DALI_LIB_DIR
        COMMAND_ERROR_IS_FATAL ANY)
string(STRIP ${DALI_LIB_DIR} DALI_LIB_DIR)

execute_process(
        COMMAND ${Python_EXECUTABLE} -c "import nvidia.dali as dali; print(\" \".join(dali.sysconfig.get_compile_flags()))"
        OUTPUT_VARIABLE DALI_COMPILE_FLAGS
        COMMAND_ERROR_IS_FATAL ANY)
string(STRIP ${DALI_COMPILE_FLAGS} DALI_COMPILE_FLAGS)

message(STATUS "Building against DALI ${DALI_VERSION} (CUDA ${DALI_CUDA_VERSION}) from ${DALI_LIB_DIR}")

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${DALI_COMPILE_FLAGS} ")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} ${DALI_COMPILE_FLAGS} ")
link_directories("${DALI_LIB_DIR}")

set(PLUGIN_SOURCES
        ${CMAKE_CURRENT_SOURCE_DIR}/naive_histogram.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/naive_histogram.cu
        )

add_library(naivehistogram SHARED ${PLUGIN_SOURCES})
target_link_libraries(naivehistogram dali)

# Used when the plugin is packaged as a Python package (see pyproject.toml): the library and
# the DALI version and flavor it was built against are installed next to the Python sources.
file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/_build_info.py"
     "DALI_VERSION = \"${DALI_VERSION}\"\nDALI_CUDA_VERSION = ${DALI_CUDA_VERSION}\n")
install(TARGETS naivehistogram LIBRARY DESTINATION dali_naive_histogram)
install(FILES "${CMAKE_CURRENT_BINARY_DIR}/_build_info.py" DESTINATION dali_naive_histogram)
