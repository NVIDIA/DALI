// Copyright (c) 2017-2018, NVIDIA CORPORATION. All rights reserved.
#include "ndll/error_handling.h"

namespace ndll {

namespace {
// Thread local string object to store error string
thread_local string g_ndll_error_string;
}  // namespace

string NDLLGetLastError() {
  string error_str = g_ndll_error_string;
  g_ndll_error_string.clear();
  return error_str;
}

void NDLLSetLastError(string error_str) {
  // Note: This currently overwrites the string if there is
  // already an error string stored.
  g_ndll_error_string = error_str;
}

}  // namespace ndll
