// Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
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

#ifndef DALI_OPERATORS_IMAGE_CONVOLUTION_GAUSSIAN_BLUR_H_
#define DALI_OPERATORS_IMAGE_CONVOLUTION_GAUSSIAN_BLUR_H_

#include <memory>
#include <vector>

#include "dali/pipeline/operator/operator.h"
#include "dali/pipeline/util/operator_impl_utils.h"

namespace dali {

#define GAUSSIAN_BLUR_SUPPORTED_TYPES \
  (uint8_t, int8_t, uint16_t, int16_t, uint32_t, int32_t, uint64_t, int64_t, float, float16)

#define GAUSSIAN_BLUR_SUPPORTED_AXES (1, 2, 3)
template <typename Backend>
class GaussianBlur : public Operator<Backend> {
 public:
  inline explicit GaussianBlur(const OpSpec& spec) : Operator<Backend>(spec) {}

  DISABLE_COPY_MOVE_ASSIGN(GaussianBlur);

 protected:
  bool CanInferOutputs() const override {
    return true;
  }

  bool SetupImpl(std::vector<OutputDesc>& output_desc, const workspace_t<Backend>& ws) override;

  void RunImpl(workspace_t<Backend>& ws) override;

 private:
  USE_OPERATOR_MEMBERS();
  std::unique_ptr<OpImplBase<Backend>> impl_;
};

}  // namespace dali

#endif  // DALI_OPERATORS_IMAGE_CONVOLUTION_GAUSSIAN_BLUR_H_
