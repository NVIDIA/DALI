// Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
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

#include <gtest/gtest.h>
#include "dali/core/error_handling.h"
#include "dali/core/cuda_utils.h"

namespace dali {

TEST(Error, EnforceFailed) {
  std::string file_and_line;
  std::string message = "Test message";
  try {
    // the two statements below must be in one line!
    file_and_line = FILE_AND_LINE; DALI_ENFORCE(!"Always fail", message);
    FAIL() << "Expected exception";
  } catch (DALIException &e) {
    std::string msg = e.what();
    EXPECT_NE(msg.find(file_and_line), std::string::npos)
      << "File/line spec not found in error `what()`, which is:\n" << msg;
    EXPECT_NE(msg.find(message), std::string::npos)
      << "Original message not found in error `what()`, which is:\n" << msg;
  } catch (...) {
    FAIL() << "Expected DALIException, got other exception";
  }
}

}  // namespace dali
