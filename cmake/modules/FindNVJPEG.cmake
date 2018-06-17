# Derived from: caffe2/cmake/Modules/FindJpegTurbo.cmake
#
# - Try to find nvjpeg
#
# The following variables are optionally searched for defaults
#  NVJPEG_ROOT_DIR:            Base directory where all NVJPEG components are found
#
# The following are set after configuration is done:
#  NVJPEG_FOUND
#  NVJPEG_INCLUDE_DIRS
#  NVJPEG_LIBRARIES
#  NVJPEG_LIBRARYRARY_DIRS

include(FindPackageHandleStandardArgs)

set(NVJPEG_ROOT_DIR "" CACHE PATH "Folder contains NVJPEG")

message(STATUS "root: ${NVJPEG_ROOT_DIR}")
find_path(NVJPEG_INCLUDE_DIR nvjpeg.h
    PATHS ${NVJPEG_ROOT_DIR}
    PATH_SUFFIXES include)

find_library(NVJPEG_LIBRARY libnvjpeg.so nvjpeg
    PATHS ${NVJPEG_ROOT_DIR}
    PATH_SUFFIXES lib lib64)

find_package_handle_standard_args(NVJPEG DEFAULT_MSG NVJPEG_INCLUDE_DIR NVJPEG_LIBRARY)

if(NVJPEG_FOUND)
  set(NVJPEG_INCLUDE_DIRS ${NVJPEG_INCLUDE_DIR})
  set(NVJPEG_LIBRARIES ${NVJPEG_LIBRARY})
  message(STATUS "Found NVJPEG    (include: ${NVJPEG_INCLUDE_DIRS}, library: ${NVJPEG_LIBRARIES})")
  mark_as_advanced(NVJPEG_ROOT_DIR NVJPEG_LIBRARY_RELEASE NVJPEG_LIBRARY_DEBUG
                   NVJPEG_LIBRARY NVJPEG_INCLUDE_DIR)
endif()
