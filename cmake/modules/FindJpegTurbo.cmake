# Source: caffe2/cmake/Modules/FindJpegTurbo.cmake
#
# - Try to find libjpeg-turbo (more specifically libturbojpeg)
#
# The following variables are optionally searched for defaults
#  JPEG_TURBO_ROOT_DIR:            Base directory where all JPEG_TURBO components are found
#
# The following are set after configuration is done:
#  JPEG_TURBO_FOUND
#  JPEG_TURBO_INCLUDE_DIRS
#  JPEG_TURBO_LIBRARIES
#  JPEG_TURBO_LIBRARYRARY_DIRS

include(FindPackageHandleStandardArgs)

set(JPEG_TURBO_ROOT_DIR "" CACHE PATH "Folder contains JPEG_TURBO")

find_path(JPEG_TURBO_INCLUDE_DIR turbojpeg.h
    PATHS ${JPEG_TURBO_ROOT_DIR}
    PATH_SUFFIXES include)

find_library(JPEG_TURBO_LIBRARY turbojpeg
    PATHS ${JPEG_TURBO_ROOT_DIR}
    PATH_SUFFIXES lib lib64)

find_package_handle_standard_args(JPEG_TURBO DEFAULT_MSG JPEG_TURBO_INCLUDE_DIR JPEG_TURBO_LIBRARY)

if(JPEG_TURBO_FOUND)
  set(JPEG_TURBO_INCLUDE_DIRS ${JPEG_TURBO_INCLUDE_DIR})
  set(JPEG_TURBO_LIBRARIES ${JPEG_TURBO_LIBRARY})
  message(STATUS "Found JPEG_TURBO    (include: ${JPEG_TURBO_INCLUDE_DIRS}, library: ${JPEG_TURBO_LIBRARIES})")
  mark_as_advanced(JPEG_TURBO_ROOT_DIR JPEG_TURBO_LIBRARY_RELEASE JPEG_TURBO_LIBRARY_DEBUG
                   JPEG_TURBO_LIBRARY JPEG_TURBO_INCLUDE_DIR)
endif()
