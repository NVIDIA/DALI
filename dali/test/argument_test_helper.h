// Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef DALI_TEST_ARGUMENT_TEST_HELPER_H_
#define DALI_TEST_ARGUMENT_TEST_HELPER_H_

#include <memory>
#include "dali/pipeline/proto/dali_proto_intern.h"

namespace dali {

// Construct and destroy generated messages in libdali, which owns the protobuf runtime.
class DLL_PUBLIC ArgumentTestHelper {
 public:
  ArgumentTestHelper();
  ~ArgumentTestHelper();
  ArgumentTestHelper(ArgumentTestHelper &&) noexcept;

  DaliProtoPriv argument();
  void TruncateExtraArgs(int size);
  void ClearExtraArgValues(int index);

 private:
  std::unique_ptr<dali_proto::Argument> proto_;
};

}  // namespace dali

#endif  // DALI_TEST_ARGUMENT_TEST_HELPER_H_
