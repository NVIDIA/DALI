// Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef DALI_OPERATORS_IMAGE_CONVOLUTION_LAPLACIAN_H_
#define DALI_OPERATORS_IMAGE_CONVOLUTION_LAPLACIAN_H_

#include <memory>
#include <vector>

#include "dali/pipeline/operator/operator.h"
#include "dali/pipeline/util/operator_impl_utils.h"
#include "dali/operators/image/convolution/laplacian_params.h"
#include "dali/operators/image/convolution/convolution_utils.h"
#include "dali/pipeline/operator/common.h"

namespace dali {

#define LAPLACIAN_CPU_SUPPORTED_TYPES \
  (uint8_t, int8_t, uint16_t, int16_t, uint32_t, int32_t, uint64_t, int64_t, float16, float)

#define LAPLACIAN_SUPPORTED_AXES (1, 2, 3)


class Laplacian : public Operator<CPUBackend> {
 public:
  inline explicit Laplacian(const OpSpec& spec)
      : Operator<CPUBackend>(spec), dtype_(spec.GetArgument<DALIDataType>("dtype")) {}

  DISABLE_COPY_MOVE_ASSIGN(Laplacian);

 protected:
  bool CanInferOutputs() const override {
    return true;
  }

  bool SetupImpl(std::vector<OutputDesc>& output_desc, const workspace_t<CPUBackend>& ws) override;

  void RunImpl(workspace_t<CPUBackend>& ws) override;

 private:
  DALIDataType dtype_;
  USE_OPERATOR_MEMBERS();
  std::unique_ptr<OpImplBase<CPUBackend>> impl_;
  DALIDataType impl_in_dtype_ = DALI_NO_TYPE;
  convolution_utils::DimDesc impl_dim_desc_ = {};
};

}  // namespace dali

#endif  // DALI_OPERATORS_IMAGE_CONVOLUTION_LAPLACIAN_H_
