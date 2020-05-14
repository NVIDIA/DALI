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

#ifndef DALI_KERNELS_ERASE_ERASE_CPU_H_
#define DALI_KERNELS_ERASE_ERASE_CPU_H_

#include <vector>
#include <utility>
#include "dali/core/common.h"
#include "dali/core/convert.h"
#include "dali/core/format.h"
#include "dali/core/error_handling.h"
#include "dali/kernels/kernel.h"
#include "dali/kernels/erase/erase_args.h"
#include "dali/kernels/common/utils.h"

namespace dali {
namespace kernels {

namespace detail {

template <typename O>
inline std::enable_if_t<std::is_trivially_copyable<O>::value, void>
CopyImpl(O *out, const O *in, size_t num) {
  std::memcpy(out, in, num * sizeof(O));
}

template <typename O>
inline std::enable_if_t<!std::is_trivially_copyable<O>::value, void>
CopyImpl(O *out, const O *in, size_t num) {
  for (size_t i = 0; i < num; ++i) {
    *out = *in;
    ++in;
    ++out;
  }
}

template <typename T, int Dims>
void EraseKernelImpl(T *data,
                     const TensorShape<Dims> &strides,
                     const TensorShape<Dims> &shape,
                     const T* fill_values,
                     int channels_dim,
                     std::integral_constant<int, 1>) {
  assert(fill_values != nullptr);
  for (int i = 0; i < shape[Dims - 1]; i++) {
    data[i] = *fill_values;
    if (channels_dim == Dims - 1) {
      fill_values++;
    }
  }
}

template <typename T, int Dims, int DimsLeft>
void EraseKernelImpl(T *data,
                     const TensorShape<Dims> &strides,
                     const TensorShape<Dims> &shape,
                     const T* fill_values,
                     int channels_dim,
                     std::integral_constant<int, DimsLeft>) {
  constexpr auto d = Dims - DimsLeft;  // NOLINT
  for (int i = 0; i < shape[d]; i++) {
    EraseKernelImpl(data, strides, shape, fill_values, channels_dim,
                    std::integral_constant<int, DimsLeft - 1>());
    data += strides[d];
    if (d == channels_dim) {
      fill_values++;
    }
  }
}

}  // namespace detail


template <typename T, int Dims>
void EraseKernel(T *data,
                 const TensorShape<Dims> &strides,
                 const TensorShape<Dims> &anchor,
                 const TensorShape<Dims> &shape,
                 const T* fill_values = nullptr,
                 int channels_dim = -1) {
  T default_fill_value = 0;
  if (fill_values == nullptr) {
    fill_values = &default_fill_value;
    assert(channels_dim == -1);
  }
  for (int d = 0; d < Dims; d++) {
    data += strides[d] * anchor[d];
  }
  if (channels_dim != -1) {
    fill_values += anchor[channels_dim];
  }
  detail::EraseKernelImpl(data, strides, shape, fill_values, channels_dim,
                          std::integral_constant<int, Dims>());
}

template <typename T, int Dims>
class EraseCpu {
 public:
  KernelRequirements Setup(KernelContext &context,
                           const InTensorCPU<T, Dims> &in,
                           const EraseArgs<T, Dims> &args) {
    KernelRequirements req;
    req.output_shapes.push_back(uniform_list_shape<Dims>(1, in.shape));
    return req;
  }

  void Run(KernelContext &context,
           OutTensorCPU<T, Dims> &out,
           const InTensorCPU<T, Dims> &in,
           const EraseArgs<T, Dims> &orig_args) {
    auto args = orig_args;
    DALI_ENFORCE(in.shape == out.shape);
    const auto &shape = out.shape;
    auto strides = GetStrides(shape);
    const T *in_ptr = in.data;
    T *out_ptr = out.data;
    if (out_ptr != in_ptr) {
      detail::CopyImpl(out_ptr, in_ptr, volume(shape));
    }

    for (auto &roi : args.rois) {
      bool valid_region = true;
      for (int d = 0; d < Dims; d++) {
        // correcting anchor if out of bounds
        if (roi.anchor[d] < 0) {
          roi.shape[d] += roi.anchor[d];
          roi.anchor[d] = 0;
        }
        // correcting shape if out of bounds
        if ((roi.anchor[d] + roi.shape[d]) > shape[d]) {
          roi.shape[d] = shape[d] - roi.anchor[d];
          assert(roi.anchor[d] + roi.shape[d] == shape[d]);
        }

        // filter out invalid regions
        if (roi.shape[d] < 1) {
          valid_region = false;
        }

        // at this point the region should be within bounds
        assert(roi.anchor[d] >= 0);
        assert(roi.anchor[d] + roi.shape[d] <= shape[d]);
      }

      if (!valid_region)
        continue;

      const T* fill_values = roi.fill_values.empty() ? nullptr : roi.fill_values.data();
      int channels_dim = -1;  // by default single-value
      int fill_values_size = roi.fill_values.size();
      if (fill_values_size > 1) {
        channels_dim = roi.channels_dim;
        DALI_ENFORCE(channels_dim >= 0 && channels_dim < Dims);
        DALI_ENFORCE(fill_values_size == in.shape[channels_dim],
          "Multi-channel fill value does not match the number of channels in the input");
        fill_values = roi.fill_values.data();
      }

      EraseKernel(out_ptr, strides, roi.anchor, roi.shape, fill_values, channels_dim);
    }
  }
};

}  // namespace kernels
}  // namespace dali

#endif  // DALI_KERNELS_ERASE_ERASE_CPU_H_

