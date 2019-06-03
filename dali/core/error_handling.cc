// Copyright (c) 2017-2018, NVIDIA CORPORATION. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "dali/core/error_handling.h"

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

void DALISetLastError(const string &error_str) {
  // Note: This currently overwrites the string if there is
  // already an error string stored.
  g_dali_error_string = error_str;
}

void DALIAppendToLastError(const string &error_str) {
  // Adds additional info to previously returned error
  g_dali_error_string += error_str;
}

void DALIReportFatalProblem(const char *file, int lineNumb, const char *pComment) {
  dali::string line = std::to_string(lineNumb);
  dali::string error_str = "[" + dali::string(file) + ":" + line + "] " + pComment;
  throw DALIException(error_str);
}


}  // namespace dali
