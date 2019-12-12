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

namespace dali {
namespace kernels {
namespace signal {
namespace dct {

namespace {

template <typename T>
void FillCosineTable(T *table, int64_t input_length, int64_t ndct, int dct_type) {
  T phase_mul = (dct_type == 1) ? M_PI / (input_length - 1) : M_PI / input_length;
  int64_t idx = 0;
  for (int64_t k = 0; k < ndct; k++) {
    T k_factor = (dct_type == 3 || dct_type == 4) ? k + T(0.5) : k;
    for (int64_t n = 0; n < input_length; n++) {
      T n_factor = (dct_type == 2 || dct_type == 4) ? n + T(0.5) : n;
      table[idx++] = std::cos(phase_mul * k_factor * n_factor);
    }
  }
}

}  // namespace

template <typename OutputType, typename InputType, int Dims>
Dct1DCpu<OutputType, InputType, Dims>::~Dct1DCpu() = default;

template <typename OutputType, typename InputType, int Dims>
KernelRequirements
Dct1DCpu<OutputType, InputType, Dims>::Setup(KernelContext &context,
                                             const InTensorCPU<InputType, Dims> &in,
                                             const DctArgs &args) {
  const auto &in_shape = in.shape;
  DALI_ENFORCE(in_shape.size() == Dims);

  args_ = args;
  args_.axis = args_.axis >= 0 ? args_.axis : Dims - 1;
  DALI_ENFORCE(args_.axis >= 0 && args_.axis < Dims,
               make_string("Axis is out of bounds: ", args_.axis));
  int64_t n = in.shape[args_.axis];

  KernelRequirements req;
  auto out_shape = in.shape;

  ScratchpadEstimator se;
  se.add<OutputType>(AllocType::Host, n * n);
  req.scratch_sizes = se.sizes;

  req.output_shapes = {TensorListShape<DynamicDimensions>({out_shape})};
  return req;
}

template <typename OutputType, typename InputType, int Dims>
void Dct1DCpu<OutputType, InputType, Dims>::Run(KernelContext &context,
                                                const OutTensorCPU<OutputType, Dims> &out,
                                                const InTensorCPU<InputType, Dims> &in,
                                                const DctArgs &args) {
  (void)args;
  assert(args_.axis >= 0 && args_.axis < Dims);
  const auto n = in.shape[args_.axis];

  assert(args_.dct_type >= 1 && args_.dct_type <= 4);

  auto cos_table_sz = n * n;
  OutputType *cos_table =
      context.scratchpad->template Allocate<OutputType>(AllocType::Host, cos_table_sz);
  memset(cos_table, 0, cos_table_sz * sizeof(OutputType));
  auto ndct = n;
  FillCosineTable(cos_table, n, ndct, args.dct_type);

  auto in_shape = in.shape;
  auto in_strides = GetStrides(in_shape);
  auto out_shape = out.shape;
  auto out_strides = GetStrides(out_shape);

  ForAxis(
    out.data, in.data, out_shape.data(), out_strides.data(), in_shape.data(), in_strides.data(),
    args_.axis, out.dim(),
    [this, &cos_table](
      OutputType *out_data, const InputType *in_data, int64_t out_size, int64_t out_stride,
      int64_t in_size, int64_t in_stride) {
        const OutputType phase_mul = M_PI / in_size;
        int64_t out_idx = 0;
        for (int64_t k = 0; k < out_size; k++) {
          int64_t in_idx = 0;
          OutputType out_val = 0;
          if (args_.dct_type == 1) {
            OutputType sign = (k % 2 == 0) ? 1 : -1;
            out_val = OutputType(0.5) * (in_data[0] + sign * in_data[(in_size - 1) * in_stride]);
          } else if (args_.dct_type == 3) {
            out_val = OutputType(0.5) * in_data[0];
          }
          const auto *cos_table_row = cos_table + k * in_size;
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

template class Dct1DCpu<double, double, 1>;
template class Dct1DCpu<double, double, 2>;
template class Dct1DCpu<double, double, 3>;

}  // namespace dct
}  // namespace signal
}  // namespace kernels
}  // namespace dali
