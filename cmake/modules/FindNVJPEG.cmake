# Derived from: caffe2/cmake/Modules/FindJpegTurbo.cmake
#
# - Try to find nvjpeg
#
# The following variables are optionally searched for defaults
#  NVJPEG_ROOT_DIR:            Base directory where all NVJPEG components are found
#
# The following are set after configuration is done:
#  NVJPEG_FOUND
#  NVJPEG_INCLUDE_DIR
#  NVJPEG_LIBRARY

set(NVJPEG_ROOT_DIR "" CACHE PATH "Folder contains NVJPEG")

find_path(NVJPEG_INCLUDE_DIR nvjpeg.h
    PATHS ${NVJPEG_ROOT_DIR} ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
    PATH_SUFFIXES include)

find_library(NVJPEG_LIBRARY libnvjpeg_static.a nvjpeg
    PATHS ${NVJPEG_ROOT_DIR} ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES}
    PATH_SUFFIXES lib lib64)


# nvJPEG 9.0 calls itself 0.1.x via API calls, and the header file doesn't tell you which
# version it is. There's not a super clean way to determine which CUDA's nvJPEG we have.
execute_process(COMMAND strings ${NVJPEG_LIBRARY} COMMAND grep /toolkit/
                COMMAND sed "s;^.*toolkit/r\\(\[^/\]\\+\\\).*$;\\1;" COMMAND sort -u
                OUTPUT_STRIP_TRAILING_WHITESPACE OUTPUT_VARIABLE NVJPEG_VERSION)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(NVJPEG
    REQUIRED_VARS NVJPEG_INCLUDE_DIR NVJPEG_LIBRARY
    VERSION_VAR NVJPEG_VERSION)

if(NVJPEG_FOUND)
  # set includes and link libs for nvJpeg

  if (POLICY CMP0075)
    cmake_policy(SET CMP0075 NEW)
  endif()

  set(CMAKE_REQUIRED_INCLUDES_OLD ${CMAKE_REQUIRED_INCLUDES_OLD})
  set(CMAKE_REQUIRED_INCLUDES ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
  set(CMAKE_REQUIRED_LIBRARIES_OLD ${CMAKE_REQUIRED_LIBRARIES})
  foreach(DIR ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES})
    list(APPEND CMAKE_REQUIRED_LIBRARIES "-L${DIR}")
  endforeach(DIR)

  list(APPEND CMAKE_REQUIRED_LIBRARIES "${NVJPEG_LIBRARY}" cudart_static culibos dl m pthread rt)
  check_cxx_symbol_exists("nvjpegCreateEx" "nvjpeg.h" NVJPEG_LIBRARY_0_2_0)
  check_cxx_symbol_exists("nvjpegBufferPinnedCreate" "nvjpeg.h" NVJPEG_DECOUPLED_API)
  check_cxx_symbol_exists("nvjpegDecodeBatchedPreAllocate" "nvjpeg.h" NVJPEG_PREALLOCATE_API)

  set(CMAKE_REQUIRED_LIBRARIES ${CMAKE_REQUIRED_LIBRARIES_OLD})
  set(CMAKE_REQUIRED_INCLUDES ${CMAKE_REQUIRED_INCLUDES_OLD})

  mark_as_advanced(NVJPEG_ROOT_DIR NVJPEG_LIBRARY_RELEASE NVJPEG_LIBRARY_DEBUG)
  message("nvJPEG found in ${NVJPEG_INCLUDE_DIR}")
  if (NVJPEG_DECOUPLED_API)
    message("nvJPEG is using new API")
  else()
    message(FATAL_ERROR "nvjpegBufferPinnedCreate is required but not present in the available version of nvJPEG")
  endif()
else()
  message("nvJPEG NOT found")
endif()
