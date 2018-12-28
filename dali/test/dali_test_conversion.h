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

#ifndef DALI_TEST_DALI_TEST_CONVERSION_H_
#define DALI_TEST_DALI_TEST_CONVERSION_H_

#include <utility>
#include <vector>
#include <string>
#include <memory>

#include "dali/util/ocv.h"
#include "dali/test/dali_test_matching.h"

namespace dali {

template <typename InputImgType, typename OutputImgType>
class GenericConversionTest : public GenericMatchingTest<InputImgType, OutputImgType> {
 protected:
  void AddDefaultArgs(OpSpec& spec) override {
    spec.AddArg("image_type", this->ImageType())
        .AddArg("output_type", this->OutputImageType());
  }

  OpSpec DefaultSchema(const string &pName, const string &pDevice = "cpu") const override {
    return OpSpec(pName)
      .AddArg("device", pDevice)
      .AddInput("input", pDevice)
      .AddOutput("output", pDevice);
  }
};

}  // namespace dali

#endif  // DALI_TEST_DALI_TEST_CONVERSION_H_
