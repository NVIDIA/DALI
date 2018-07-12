# Try to find the LMBD libraries and headers
#  LMDB_FOUND - system has LMDB lib
#  LMDB_INCLUDE_DIR - the LMDB include directory
#  LMDB_LIBRARIES - Libraries needed to use LMDB

# FindCWD based on FindGMP by:
# Copyright (c) 2006, Laurent Montel, <montel@kde.org>
#
# Redistribution and use is allowed according to the terms of the BSD license.

# Adapted from FindCWD by:
# Copyright 2013 Conrad Steenberg <conrad.steenberg@gmail.com>
# Aug 31, 2013

if(MSVC)
  find_package(LMDB NO_MODULE)
else()
  find_path(LMDB_INCLUDE_DIR NAMES  lmdb.h PATHS "$ENV{LMDB_DIR}/include")
  find_library(LMDB_LIBRARIES NAMES lmdb   PATHS "$ENV{LMDB_DIR}/lib" )
endif()

if(LMDB_INCLUDE_DIR)
  # LMBD doesn't use pkg-config file so we need to parse it header to get version
  caffe_parse_header(${LMDB_INCLUDE_DIR}/lmdb.h
                     LMDB_VERSION_LINES MDB_VERSION_MAJOR MDB_VERSION_MINOR MDB_VERSION_PATCH)
  set(LMDB_VERSION "${MDB_VERSION_MAJOR}.${MDB_VERSION_MINOR}.${MDB_VERSION_PATCH}")
endif()

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(LMDB
    REQUIRED_VARS LMDB_INCLUDE_DIR LMDB_LIBRARIES
    VERSION_VAR LMDB_VERSION)
