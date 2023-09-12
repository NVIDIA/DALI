// Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
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

#ifndef DALI_KERNELS_SIGNAL_WAVELET_WAVELET_GPU_CUH_
#define DALI_KERNELS_SIGNAL_WAVELET_WAVELET_GPU_CUH_

#include <memory>
#include <string>
#include <vector>
#include "dali/core/common.h"
#include "dali/core/error_handling.h"
#include "dali/core/format.h"
#include "dali/core/util.h"
#include "dali/kernels/kernel.h"
#include "dali/kernels/signal/wavelet/mother_wavelet.cuh"

// makes sure both tensors have the same number of samples and
// that they're one-dimensional
#define ENFORCE_SHAPES(a_shape, b_shape)                                   \
  do {                                                                     \
    DALI_ENFORCE(a_shape.num_samples() == b_shape.num_samples(),           \
                 "a and b tensors must have the same amount of samples."); \
    for (int i = 0; i < a_shape.num_samples(); ++i) {                      \
      DALI_ENFORCE(a_shape.tensor_shape(i).size() == 1,                    \
                   "Tensor of a coeffs should be 1-dimensional.");         \
      DALI_ENFORCE(b_shape.tensor_shape(i).size() == 1,                    \
                   "Tensor of b coeffs should be 1-dimensional.");         \
    }                                                                      \
  } while (0);

namespace dali {
namespace kernels {
namespace signal {

// stores data needed to reconstruct wavelet input arguments
template <typename T = float>
struct WaveletSpan {
  // lower limit of wavelet samples
  T begin = -1.0;

  // upper limit of wavelet samples
  T end = 1.0;

  // wavelet sampling rate (samples/s)
  T sampling_rate = 1000.0;
};

template class WaveletSpan<float>;
template class WaveletSpan<double>;

template <typename T>
struct SampleDesc {
  const T *a = nullptr;
  int64_t size_a = 0;
  const T *b = nullptr;
  int64_t size_b = 0;
  T *in = nullptr;
  int64_t size_in = 0;
  T *out = nullptr;
  WaveletSpan<T> span;
};

template <typename T, template <typename> class W>
class DLL_PUBLIC WaveletGpu {
 public:
  static_assert(std::is_floating_point<T>::value, "Only floating point types are supported");

  DLL_PUBLIC WaveletGpu() = default;
  DLL_PUBLIC ~WaveletGpu() = default;

  DLL_PUBLIC KernelRequirements Setup(KernelContext &context, const InListGPU<T> &a,
                                      const InListGPU<T> &b, const WaveletSpan<T> &span,
                                      const std::vector<T> &args);

  DLL_PUBLIC void Run(KernelContext &ctx, OutListGPU<T> &out, const InListGPU<T> &a,
                      const InListGPU<T> &b, const WaveletSpan<T> &span);

  static TensorListShape<> GetOutputShape(const TensorListShape<> &a_shape,
                                          const TensorListShape<> &b_shape,
                                          const WaveletSpan<T> &span);

 private:
  W<T> wavelet_;
};

}  // namespace signal
}  // namespace kernels
}  // namespace dali

#endif  // DALI_KERNELS_SIGNAL_WAVELET_WAVELET_GPU_CUH_
