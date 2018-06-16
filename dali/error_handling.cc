// Copyright (c) 2017-2018, NVIDIA CORPORATION. All rights reserved.
#include "dali/error_handling.h"

namespace dali {

namespace {
// Thread local string object to store error string
thread_local string g_dali_error_string;
}  // namespace

string DALIGetLastError() {
  string error_str = g_dali_error_string;
  g_dali_error_string.clear();
  return error_str;
}

void DALISetLastError(string error_str) {
  // Note: This currently overwrites the string if there is
  // already an error string stored.
  g_dali_error_string = error_str;
}

void DALIReportFatalProblem(const char *file, int lineNumb, const char *pComment) {
  dali::string line = std::to_string(lineNumb);
  dali::string error_str = "[" + dali::string(file) + ":" + line + "] " + pComment;
  throw std::runtime_error(error_str);
}

}  // namespace dali
