# Source: caffe2/cmake/Modules/FindJpegTurbo.cmake
#
# - Try to find libjpeg-turbo (more specifically libturbojpeg)
#
# The following variables are optionally searched for defaults
#  JPEG_TURBO_ROOT_DIR:         Base directory where all JPEG_TURBO components are found
#
# The following are set after configuration is done:
#  JpegTurbo_FOUND
#  JpegTurbo_INCLUDE_DIR
#  JpegTurbo_LIBRARY
#  JpegTurbo_VERSION

set(JPEG_TURBO_ROOT_DIR "" CACHE PATH "Folder contains JpegTurbo")

find_package(PkgConfig QUIET)
if (PkgConfig_FOUND)
    # use pkg-config only if it has been found, if it fails it fall back to
    # JPEG_TURBO_ROOT_DIR hint
    pkg_check_modules(JpegTurbo QUIET libturbojpeg)
endif()

find_path(JpegTurbo_INCLUDE_DIR turbojpeg.h
    PATHS ${JPEG_TURBO_ROOT_DIR} ${JpegTurbo_INCLUDE_DIRS}
    PATH_SUFFIXES include)

find_library(JpegTurbo_LIBRARY libturbojpeg.so turbojpeg
    PATHS ${JPEG_TURBO_ROOT_DIR} ${JpegTurbo_LIBRARY_DIRS}
    PATH_SUFFIXES lib lib64)

# if pkg-config has no info about libjpeg-turbo find and parse jconfig.h
if (NOT DEFINED JpegTurbo_VERSION)
    find_path(JpegTurbo_CONFIG_INCLUDE_DIR jconfig.h
        PATHS ${JPEG_TURBO_ROOT_DIR} ${JpegTurbo_INCLUDE_DIRS}
        PATH_SUFFIXES include)
    file(STRINGS "${JpegTurbo_CONFIG_INCLUDE_DIR}/jconfig.h" JpegTurbo_VERSION REGEX "^#define LIBJPEG_TURBO_VERSION[ ]+[0-9]+\.[0-9]+\.[0-9]+")
    string(REGEX REPLACE "^#define LIBJPEG_TURBO_VERSION[ ]+([^\"]*).*" "\\1" JpegTurbo_VERSION "${JpegTurbo_VERSION}")
endif()

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(JpegTurbo
    REQUIRED_VARS JpegTurbo_INCLUDE_DIR JpegTurbo_LIBRARY
    VERSION_VAR JpegTurbo_VERSION)

if(JpegTurbo_FOUND)
  mark_as_advanced(JPEG_TURBO_ROOT_DIR JpegTurbo_LIBRARY_RELEASE JpegTurbo_LIBRARY_DEBUG)
endif()
