// Copyright (c) 2020-2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "dali/operators/image/convolution/convolution_utils.h"
#include "dali/operators/image/convolution/gaussian_blur_params.h"
#include "dali/pipeline/operator/common.h"
#include "dali/pipeline/operator/operator.h"
#include "dali/pipeline/operator/sequence_operator.h"
#include "dali/pipeline/util/operator_impl_utils.h"

namespace dali {

#define GAUSSIAN_BLUR_CPU_SUPPORTED_TYPES \
  (uint8_t, int8_t, uint16_t, int16_t, uint32_t, int32_t, uint64_t, int64_t, float16, float, double)

// TODO(klecki): float16 support - it's not easily compatible with float window,
// need to introduce some cast in between and expose it in the kernels
#define GAUSSIAN_BLUR_GPU_SUPPORTED_TYPES \
  (uint8_t, int8_t, uint16_t, int16_t, uint32_t, int32_t, uint64_t, int64_t, float, double)

#define GAUSSIAN_BLUR_SUPPORTED_AXES (1, 2, 3)

template <typename Backend>
class GaussianBlur : public SequenceOperator<Backend> {
 public:
  inline explicit GaussianBlur(const OpSpec& spec)
      : SequenceOperator<Backend>(spec) {
    spec.TryGetArgument(dtype_, "dtype");
  }

  DISABLE_COPY_MOVE_ASSIGN(GaussianBlur);

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

namespace gaussian_blur {

constexpr static const char* kSigmaArgName = "sigma";
constexpr static const char* kWindowSizeArgName = "window_size";

/**
 * @brief Obtain the parameters needed for generating Gaussian Windows for GaussianBlur Operator.
 */
template <int axes>
inline GaussianBlurParams<axes> ObtainSampleParams(int sample, const OpSpec& spec,
                                                   const ArgumentWorkspace& ws) {
  GaussianBlurParams<axes> params;
  GetGeneralizedArg<float>(make_span(params.sigmas), kSigmaArgName, sample, spec, ws);
  GetGeneralizedArg<int>(make_span(params.window_sizes), kWindowSizeArgName, sample, spec, ws);
  for (int i = 0; i < axes; i++) {
    DALI_ENFORCE(
        !(params.sigmas[i] == 0 && params.window_sizes[i] == 0),
        make_string("`sigma` and `window_size` shouldn't be 0 at the same time for sample: ",
                    sample, ", axis: ", i, "."));
    DALI_ENFORCE(params.sigmas[i] >= 0,
                 make_string("`sigma` must have non-negative values, got ", params.sigmas[i],
                             " for sample: ", sample, ", axis: ", i, "."));
    DALI_ENFORCE(params.window_sizes[i] >= 0,
                 make_string("`window_size` must have non-negative values, got ",
                             params.window_sizes[i], " for sample: ", sample, ", axis : ", i, "."));
    if (params.window_sizes[i] == 0) {
      params.window_sizes[i] = SigmaToDiameter(params.sigmas[i]);
    } else if (params.sigmas[i] == 0.f) {
      params.sigmas[i] = DiameterToSigma(params.window_sizes[i]);
    }
  }
  return params;
}

}  // namespace gaussian_blur
}  // namespace dali

#endif  // DALI_OPERATORS_IMAGE_CONVOLUTION_GAUSSIAN_BLUR_H_
