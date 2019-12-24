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

template <typename T, int Dims>
void EraseKernelImpl(T *data,
                     const TensorShape<Dims> &strides,
                     const TensorShape<Dims> &shape,
                     std::integral_constant<int, 1>) {
  for (int i = 0; i < shape[Dims - 1]; i++) {
    data[i] = T(0);  // TODO(janton): support fill value
  }
}

template <typename T, int Dims, int DimsLeft>
void EraseKernelImpl(T *data,
                     const TensorShape<Dims> &strides,
                     const TensorShape<Dims> &shape,
                     std::integral_constant<int, DimsLeft>) {
  constexpr auto d = Dims - DimsLeft;  // NOLINT
  for (int i = 0; i < shape[d]; i++) {
    EraseKernelImpl(data, strides, shape,
                    std::integral_constant<int, DimsLeft - 1>());
    data += strides[d];
  }
}

}  // namespace detail


template <typename T, int Dims>
void EraseKernel(T *data,
                 const TensorShape<Dims> &strides,
                 const TensorShape<Dims> &anchor,
                 const TensorShape<Dims> &shape) {
  for (int d = 0; d < Dims; d++) {
    data += strides[d] * anchor[d];
  }
  detail::EraseKernelImpl(data, strides, shape,
                          std::integral_constant<int, Dims>());
}

template <typename T, int Dims>
class EraseCpu {
 public:
  KernelRequirements Setup(KernelContext &context,
                           const InTensorCPU<T, Dims> &in,
                           const EraseArgs<Dims> &args) {
    KernelRequirements req;
    req.output_shapes.push_back(uniform_list_shape<Dims>(1, in.shape));
    return req;
  }

  void Run(KernelContext &context,
           OutTensorCPU<T, Dims> &out,
           const InTensorCPU<T, Dims> &in,
           const EraseArgs<Dims> &args) {
    DALI_ENFORCE(in.shape == out.shape);
    const auto &shape = out.shape;
    auto strides = GetStrides(shape);
    const T *in_ptr = in.data;
    T *out_ptr = out.data;
    if (out_ptr != in_ptr) {
      std::memcpy(out_ptr, in_ptr, volume(shape) * sizeof(T));
    }

    for (auto &roi : args.rois) {
      for (int d = 0; d < Dims; d++) {
        DALI_ENFORCE(roi.anchor[d]>=0 && (roi.anchor[d] + roi.shape[d]) <= shape[d],
          make_string("Erase region-of-interest is out of bounds: dimension=", d, 
            " roi_anchor=", roi.anchor[d], " roi_shape=", roi.shape[d], " data_shape=", shape[d]));
      }
      EraseKernel(out_ptr, strides, roi.anchor, roi.shape);
    }
  }
};

}  // namespace kernels
}  // namespace dali

#endif  // DALI_KERNELS_ERASE_ERASE_CPU_H_