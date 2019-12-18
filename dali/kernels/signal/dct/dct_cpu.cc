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
void FillCosineTableTypeI(T *table, int64_t input_length, int64_t ndct, bool normalize) {
  assert(input_length > 1);
  assert(!normalize);
  double phase_mul = M_PI / (input_length - 1);
  int64_t idx = 0;
  for (int64_t k = 0; k < ndct; k++) {
    table[idx++] = 0.5;  // n = 0
    for (int64_t n = 1; n < input_length-1; n++) {
      table[idx++] = std::cos(phase_mul * k * n);
    }
    table[idx++] = k % 2 == 0 ?  0.5 : -0.5;  // n = input_length - 1
  }
}

template <typename T>
void FillCosineTableTypeII(T *table, int64_t input_length, int64_t ndct, bool normalize) {
  double phase_mul = M_PI / input_length;
  double factor_k_0 = 1, factor_k_i = 1;
  if (normalize) {
    factor_k_i = std::sqrt(2.0 / input_length);
    factor_k_0 = 1.0 / std::sqrt(input_length);
  }
  int64_t idx = 0;
  for (int64_t k = 0; k < ndct; k++) {
    double norm_factor = (k == 0) ? factor_k_0 : factor_k_i;
    for (int64_t n = 0; n < input_length; n++) {
      table[idx++] = norm_factor * std::cos(phase_mul * (n + 0.5) * k);
    }
  }
}


template <typename T>
void FillCosineTableTypeIII(T *table, int64_t input_length, int64_t ndct, bool normalize) {
  double phase_mul = M_PI / input_length;
  double factor_n_0 = 0.5, factor_n_i = 1;
  if (normalize) {
    factor_n_i = std::sqrt(2.0 / input_length);
    factor_n_0 = 1.0 / std::sqrt(input_length);
  }
  int64_t idx = 0;
  for (int64_t k = 0; k < ndct; k++) {
    table[idx++] = factor_n_0;  // n = 0
    for (int64_t n = 1; n < input_length; n++) {
      table[idx++] = factor_n_i * std::cos(phase_mul * n * (k + 0.5));
    }
  }
}


template <typename T>
void FillCosineTableTypeIV(T *table, int64_t input_length, int64_t ndct, bool normalize) {
  double phase_mul = M_PI / input_length;
  double factor = normalize ? std::sqrt(2.0 / input_length) : 1.0;
  int64_t idx = 0;
  for (int64_t k = 0; k < ndct; k++) {
    for (int64_t n = 0; n < input_length; n++) {
      table[idx++] = factor * std::cos(phase_mul * (n + 0.5) * (k + 0.5));
    }
  }
}


template <typename T>
void FillCosineTable(T *table, int64_t input_length, int64_t ndct, int dct_type, bool normalize) {
  switch (dct_type) {
    case 1:
      FillCosineTableTypeI(table, input_length, ndct, normalize);
      break;
    case 2:
      FillCosineTableTypeII(table, input_length, ndct, normalize);
      break;
    case 3:
      FillCosineTableTypeIII(table, input_length, ndct, normalize);
      break;
    case 4:
      FillCosineTableTypeIV(table, input_length, ndct, normalize);
      break;
    default:
      assert(false);
  }
}

}  // namespace

template <typename OutputType, typename InputType, int Dims>
Dct1DCpu<OutputType, InputType, Dims>::~Dct1DCpu() = default;

template <typename OutputType, typename InputType, int Dims>
KernelRequirements
Dct1DCpu<OutputType, InputType, Dims>::Setup(KernelContext &context,
                                             const InTensorCPU<InputType, Dims> &in,
                                             const DctArgs &original_args) {
  const auto &in_shape = in.shape;
  DALI_ENFORCE(in_shape.size() == Dims);

  auto args = original_args;
  args.axis = args.axis >= 0 ? args.axis : Dims - 1;
  DALI_ENFORCE(args.axis >= 0 && args.axis < Dims,
               make_string("Axis is out of bounds: ", args.axis));
  int64_t n = in.shape[args.axis];

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
  out_shape[args.axis] = args.ndct;

  if (cos_table_.empty() || args != args_) {
    auto cos_table_sz = n * args.ndct;
    cos_table_.resize(cos_table_sz);
    FillCosineTable(cos_table_.data(), n, args.ndct, args.dct_type, args.normalize);
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
                                                const DctArgs &args) {
  (void)args;
  assert(args_.axis >= 0 && args_.axis < Dims);
  const auto n = in.shape[args_.axis];

  assert(args_.dct_type >= 1 && args_.dct_type <= 4);

  auto in_shape = in.shape;
  auto in_strides = GetStrides(in_shape);
  auto out_shape = out.shape;
  auto out_strides = GetStrides(out_shape);

  ForAxis(
    out.data, in.data, out_shape.data(), out_strides.data(), in_shape.data(), in_strides.data(),
    args_.axis, out.dim(),
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
