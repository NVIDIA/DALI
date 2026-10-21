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

#include "dali/test/argument_test_helper.h"
#include "dali/pipeline/dali.pb.h"

namespace dali {

ArgumentTestHelper::ArgumentTestHelper() : proto_(std::make_unique<dali_proto::Argument>()) {}
ArgumentTestHelper::~ArgumentTestHelper() = default;
ArgumentTestHelper::ArgumentTestHelper(ArgumentTestHelper &&) noexcept = default;

DaliProtoPriv ArgumentTestHelper::argument() {
  return DaliProtoPriv(proto_.get());
}

void ArgumentTestHelper::TruncateExtraArgs(int size) {
  while (proto_->extra_args_size() > size)
    proto_->mutable_extra_args()->RemoveLast();
}

void ArgumentTestHelper::ClearExtraArgValues(int index) {
  auto *field = proto_->mutable_extra_args(index);
  field->clear_ints();
  field->clear_floats();
  field->clear_bools();
  field->clear_strings();
}

}  // namespace dali
