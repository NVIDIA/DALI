// Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
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

#include "dali/kernels/audio/mel_scale/mel_filter_bank_cpu.h"
#include <cmath>
#include <complex>
#include <vector>
#include "dali/core/common.h"
#include "dali/core/error_handling.h"
#include "dali/core/format.h"
#include "dali/kernels/kernel.h"
#include "dali/kernels/common/for_axis.h"
#include "dali/kernels/common/utils.h"

namespace dali {
namespace kernels {
namespace audio {

template <typename T, int Dims>
MelFilterBankCpu<T, Dims>::~MelFilterBankCpu() = default;

template <typename T, int Dims>
KernelRequirements MelFilterBankCpu<T, Dims>::Setup(
    KernelContext &context,
    const InTensorCPU<T, Dims> &in,
    const MelFilterBankArgs &args) {
  auto axis = args.axis >= 0 ? args.axis : Dims - 2;
  auto nfft = args.nfft > 0 ? args.nfft : 2 * (in.shape[axis] - 1);
  DALI_ENFORCE(axis == Dims - 2,
    "Input is expected to be a spectrogram with the last two dimensions being FFT bin index and "
    "window index respectively");

  auto out_shape = in.shape;
  out_shape[axis] = args.nfilter;
  std::vector<TensorShape<DynamicDimensions>> tmp = {out_shape};  // workaround for clang-6 bug
  KernelRequirements req;
  req.output_shapes = {TensorListShape<DynamicDimensions>(tmp)};

  if (!impl_ || args_ != args) {
    args_ = args;
    impl_.reset(new MelFilterBankCpuImpl<T>(args.nfilter, nfft, args.sample_rate));
  }

  return req;
}

template <typename T, int Dims>
void MelFilterBankCpu<T, Dims>::Run(
    KernelContext &context,
    const OutTensorCPU<T, Dims> &out,
    const InTensorCPU<T, Dims> &in,
    const MelFilterBankArgs &args) {
  DALI_ENFORCE(impl_ != nullptr);
  auto in_shape = in.shape;
  auto nwin = in_shape[Dims - 1];
  auto in_strides = GetStrides(in_shape);
  auto out_shape = out.shape;
  auto out_strides = GetStrides(out_shape);

  auto axis = args.axis >= 0 ? args.axis : Dims - 2;  // vertical dim represents FFT space
  auto for_axis_ndim = out.dim() - 1;  // squeeze last dim
  ForAxis(
    out.data, in.data, out_shape.data(), out_strides.data(), in_shape.data(), in_strides.data(),
    axis, for_axis_ndim,
    [this, nwin](
        T *out_data, const T *in_data,
        int64_t out_size, int64_t out_stride, int64_t in_size, int64_t in_stride) {
      impl_->Compute(out_data, in_data, nwin);
    });
}

template class MelFilterBankCpu<float, 2>;
template class MelFilterBankCpu<double, 2>;

template class MelFilterBankCpu<float, 3>;
template class MelFilterBankCpu<double, 3>;

template class MelFilterBankCpu<float, 4>;
template class MelFilterBankCpu<double, 4>;

}  // namespace audio
}  // namespace kernels
}  // namespace dali
