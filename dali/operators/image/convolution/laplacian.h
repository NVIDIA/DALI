// Copyright (c) 2021-2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "dali/operators/image/convolution/convolution_utils.h"
#include "dali/operators/image/convolution/laplacian_params.h"
#include "dali/pipeline/operator/common.h"
#include "dali/pipeline/operator/operator.h"
#include "dali/pipeline/operator/sequence_operator.h"
#include "dali/pipeline/util/operator_impl_utils.h"

namespace dali {

#define LAPLACIAN_CPU_SUPPORTED_TYPES \
  (uint8_t, int8_t, uint16_t, int16_t, uint32_t, int32_t, uint64_t, int64_t, float16, float)

// TODO(klecki): float16 support - it's not easily compatible with float window,
// need to introduce some cast in between and expose it in the kernels
#define LAPLACIAN_GPU_SUPPORTED_TYPES \
  (uint8_t, int8_t, uint16_t, int16_t, uint32_t, int32_t, uint64_t, int64_t, float)

#define LAPLACIAN_SUPPORTED_AXES (1, 2, 3)


template <typename Backend>
class Laplacian : public SequenceOperator<Backend> {
 public:
  inline explicit Laplacian(const OpSpec& spec)
      : SequenceOperator<Backend>(spec) {
    spec.TryGetArgument(dtype_, "dtype");
  }

  DISABLE_COPY_MOVE_ASSIGN(Laplacian);

 protected:
  bool CanInferOutputs() const override {
    return true;
  }

  bool ShouldExpandChannels(int input_idx) const override {
    (void)input_idx;
    return true;
  }

  bool ShouldExpand(const workspace_t<Backend>& ws) override;

  // Overrides unnecessary coalescing
  bool ProcessOutputDesc(std::vector<OutputDesc>& output_desc, const workspace_t<Backend>& ws,
                         bool is_inferred) override {
    assert(is_inferred && output_desc.size() == 1);
    const auto& input = ws.template Input<Backend>(0);
    // The shape of data stays untouched
    output_desc[0].shape = input.shape();
    return true;
  }

  bool SetupImpl(std::vector<OutputDesc>& output_desc, const workspace_t<Backend>& ws) override;

  void RunImpl(workspace_t<Backend>& ws) override;

 private:
  DALIDataType dtype_ = DALI_NO_TYPE;
  USE_OPERATOR_MEMBERS();
  std::unique_ptr<OpImplBase<Backend>> impl_;
  DALIDataType impl_in_dtype_ = DALI_NO_TYPE;
  convolution_utils::DimDesc dim_desc_ = {};
  convolution_utils::DimDesc impl_dim_desc_ = {};
};

}  // namespace dali

#endif  // DALI_OPERATORS_IMAGE_CONVOLUTION_LAPLACIAN_H_
