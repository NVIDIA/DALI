// Copyright (c) 2020-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <functional>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "dali/core/static_switch.h"
#include "dali/kernels/imgproc/convolution/separable_convolution_gpu.h"
#include "dali/kernels/kernel_manager.h"
#include "dali/operators/image/convolution/gaussian_blur.h"
#include "dali/operators/image/convolution/gaussian_blur_params.h"
#include "dali/pipeline/data/views.h"
#include "dali/pipeline/operator/common.h"
#include "dali/operators/image/convolution/gaussian_blur_gpu.h"

namespace dali {

using namespace gaussian_blur;  // NOLINT

namespace gaussian_blur {

// Functions below are explicitly instantiated in separate compilation unit to allow
// for parallel compitlation of underlying kernels.

extern template op_impl_uptr GetGaussianBlurGpuImpl<uint8_t, uint8_t>(const OpSpec*, DimDesc);
extern template op_impl_uptr GetGaussianBlurGpuImpl<float, uint8_t>(const OpSpec*, DimDesc);

extern template op_impl_uptr GetGaussianBlurGpuImpl<int8_t, int8_t>(const OpSpec*, DimDesc);
extern template op_impl_uptr GetGaussianBlurGpuImpl<float, int8_t>(const OpSpec*, DimDesc);

extern template op_impl_uptr GetGaussianBlurGpuImpl<uint16_t, uint16_t>(const OpSpec*, DimDesc);
extern template op_impl_uptr GetGaussianBlurGpuImpl<float, uint16_t>(const OpSpec*, DimDesc);

extern template op_impl_uptr GetGaussianBlurGpuImpl<int16_t, int16_t>(const OpSpec*, DimDesc);
extern template op_impl_uptr GetGaussianBlurGpuImpl<float, int16_t>(const OpSpec*, DimDesc);

extern template op_impl_uptr GetGaussianBlurGpuImpl<uint32_t, uint32_t>(const OpSpec*, DimDesc);
extern template op_impl_uptr GetGaussianBlurGpuImpl<float, uint32_t>(const OpSpec*, DimDesc);

extern template op_impl_uptr GetGaussianBlurGpuImpl<int32_t, int32_t>(const OpSpec*, DimDesc);
extern template op_impl_uptr GetGaussianBlurGpuImpl<float, int32_t>(const OpSpec*, DimDesc);

extern template op_impl_uptr GetGaussianBlurGpuImpl<uint64_t, uint64_t>(const OpSpec*, DimDesc);
extern template op_impl_uptr GetGaussianBlurGpuImpl<float, uint64_t>(const OpSpec*, DimDesc);

extern template op_impl_uptr GetGaussianBlurGpuImpl<int64_t, int64_t>(const OpSpec*, DimDesc);
extern template op_impl_uptr GetGaussianBlurGpuImpl<float, int64_t>(const OpSpec*, DimDesc);

extern template op_impl_uptr GetGaussianBlurGpuImpl<float16, float16>(const OpSpec*, DimDesc);
extern template op_impl_uptr GetGaussianBlurGpuImpl<float, float16>(const OpSpec*, DimDesc);

extern template op_impl_uptr GetGaussianBlurGpuImpl<float, float>(const OpSpec*, DimDesc);

extern template op_impl_uptr GetGaussianBlurGpuImpl<double, double>(const OpSpec*, DimDesc);
extern template op_impl_uptr GetGaussianBlurGpuImpl<float, double>(const OpSpec*, DimDesc);

}  // namespace gaussian_blur

// Passing to the kernel less samples (not split into frames) speeds-up
// the processing, so expand frames dim only if some argument was specified per-frame
template <>
bool GaussianBlur<GPUBackend>::ShouldExpand(const Workspace &ws) {
  const auto& input = ws.Input<GPUBackend>(0);
  auto layout = input.GetLayout();
  dim_desc_ = convolution_utils::ParseAndValidateDim(input.shape().sample_dim(), layout);
  bool should_expand = Base::ShouldExpand(ws) && HasPerFrameArgInputs(ws);
  if (should_expand) {
    assert(dim_desc_.usable_axes_start > 0);
    dim_desc_.total_axes_count -= dim_desc_.usable_axes_start;
    dim_desc_.usable_axes_start = 0;
  }
  return should_expand;
}

template <>
bool GaussianBlur<GPUBackend>::SetupImpl(std::vector<OutputDesc>& output_desc,
                                         const Workspace &ws) {
  const auto& input = ws.Input<GPUBackend>(0);
  assert(input.GetLayout().empty() || input.GetLayout().size() == dim_desc_.total_axes_count);
  auto dtype = dtype_ == DALI_NO_TYPE ? input.type() : dtype_;
  DALI_ENFORCE(dtype == input.type() || dtype == DALI_FLOAT,
               "Output data type must be same as input, FLOAT or skipped (defaults to input type)");

  if (!impl_ || impl_in_dtype_ != input.type() || impl_dim_desc_ != dim_desc_) {
    impl_in_dtype_ = input.type();
    impl_dim_desc_ = dim_desc_;

    // clang-format off
    TYPE_SWITCH(input.type(), type2id, In, GAUSSIAN_BLUR_GPU_SUPPORTED_TYPES, (
      if (dtype == input.type()) {
        impl_ = GetGaussianBlurGpuImpl<In, In>(&spec_, dim_desc_);
      } else {
        impl_ = GetGaussianBlurGpuImpl<float, In>(&spec_, dim_desc_);
      }
    ), DALI_FAIL(make_string("Unsupported data type: ", input.type())));  // NOLINT
    // clang-format on
  }

  return impl_->SetupImpl(output_desc, ws);
}

template <>
void GaussianBlur<GPUBackend>::RunImpl(Workspace &ws) {
  impl_->RunImpl(ws);
}

DALI_REGISTER_OPERATOR(GaussianBlur, GaussianBlur<GPUBackend>, GPU);

}  // namespace dali
