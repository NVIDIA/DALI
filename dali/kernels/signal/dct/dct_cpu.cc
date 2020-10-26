// Copyright (c) 2019-2020, NVIDIA CORPORATION. All rights reserved.
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

#include "dali/kernels/signal/dct/dct_cpu.h"
#include <cmath>
#include "dali/core/common.h"
#include "dali/core/convert.h"
#include "dali/core/error_handling.h"
#include "dali/core/format.h"
#include "dali/core/util.h"
#include "dali/kernels/common/for_axis.h"
#include "dali/kernels/common/utils.h"
#include "dali/kernels/kernel.h"
#include "dali/kernels/signal/dct/table.h"

namespace dali {
namespace kernels {
namespace signal {
namespace dct {

template <typename OutputType, typename InputType, int Dims>
Dct1DCpu<OutputType, InputType, Dims>::~Dct1DCpu() = default;

template <typename OutputType, typename InputType, int Dims>
KernelRequirements
Dct1DCpu<OutputType, InputType, Dims>::Setup(KernelContext &context,
                                             const InTensorCPU<InputType, Dims> &in,
                                             const DctArgs &original_args, int axis) {
  const auto &in_shape = in.shape;
  DALI_ENFORCE(in_shape.size() == Dims);

  auto args = original_args;
  axis_ = axis >= 0 ? axis : Dims - 1;
  DALI_ENFORCE(axis_ >= 0 && axis_ < Dims,
               make_string("Axis is out of bounds: ", axis_));
  int64_t n = in.shape[axis_];

  if (args.dct_type == 1) {
    DALI_ENFORCE(n > 1, "DCT type I requires an input length > 1");
    if (args.normalize) {
      DALI_WARN("DCT type-I does not support orthogonal normalization. Ignoring");
      args.normalize = false;
    }
  }

  if (args.ndct <= 0 || args.ndct > n) {
    args.ndct = n;
  }

  auto out_shape = in.shape;
  out_shape[axis_] = args.ndct;

  size_t cos_table_sz = n * args.ndct;
  if (cos_table_.size() != cos_table_sz || args != args_) {
    cos_table_.resize(cos_table_sz);
    FillCosineTable(cos_table_.data(), n, args);
    args_ = args;
  }

  KernelRequirements req;
  req.output_shapes = {TensorListShape<DynamicDimensions>({out_shape})};
  return req;
}

template <typename OutputType, typename InputType, int Dims>
void Dct1DCpu<OutputType, InputType, Dims>::Run(KernelContext &context,
                                                const OutTensorCPU<OutputType, Dims> &out,
                                                const InTensorCPU<InputType, Dims> &in,
                                                const DctArgs &args, int axis) {
  (void)args;
  (void)axis;
  assert(axis_ >= 0 && axis_ < Dims);
  const auto n = in.shape[axis_];

  assert(args_.dct_type >= 1 && args_.dct_type <= 4);

  auto in_shape = in.shape;
  auto in_strides = GetStrides(in_shape);
  auto out_shape = out.shape;
  auto out_strides = GetStrides(out_shape);

  ForAxis(
    out.data, in.data, out_shape.data(), out_strides.data(), in_shape.data(), in_strides.data(),
    axis_, out.dim(),
    [this](
      OutputType *out_data, const InputType *in_data, int64_t out_size, int64_t out_stride,
      int64_t in_size, int64_t in_stride) {
        int64_t out_idx = 0;
        for (int64_t k = 0; k < out_size; k++) {
          OutputType out_val = 0;
          const auto *cos_table_row = cos_table_.data() + k * in_size;
          int64_t in_idx = 0;
          for (int64_t n = 0; n < in_size; n++) {
            OutputType in_val = in_data[in_idx];
            in_idx += in_stride;
            out_val += in_val * cos_table_row[n];
          }
          out_data[out_idx] = out_val;
          out_idx += out_stride;
        }
    });
}

template class Dct1DCpu<float, float, 1>;
template class Dct1DCpu<float, float, 2>;
template class Dct1DCpu<float, float, 3>;
template class Dct1DCpu<float, float, 4>;

template class Dct1DCpu<double, double, 1>;
template class Dct1DCpu<double, double, 2>;
template class Dct1DCpu<double, double, 3>;
template class Dct1DCpu<double, double, 4>;

}  // namespace dct
}  // namespace signal
}  // namespace kernels
}  // namespace dali
