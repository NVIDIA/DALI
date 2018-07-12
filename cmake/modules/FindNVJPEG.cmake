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
    PATHS ${NVJPEG_ROOT_DIR}
    PATH_SUFFIXES include)

find_library(NVJPEG_LIBRARY libnvjpeg_static.a nvjpeg
    PATHS ${NVJPEG_ROOT_DIR}
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
  mark_as_advanced(NVJPEG_ROOT_DIR NVJPEG_LIBRARY_RELEASE NVJPEG_LIBRARY_DEBUG)
endif()
