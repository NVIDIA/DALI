// Copyright (c) 2019-2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef DALI_OPERATORS_MATH_EXPRESSIONS_EXPRESSION_IMPL_CPU_H_
#define DALI_OPERATORS_MATH_EXPRESSIONS_EXPRESSION_IMPL_CPU_H_

#include <vector>

#include "dali/pipeline/data/types.h"
#include "dali/operators/math/expressions/arithmetic_meta.h"
#include "dali/operators/math/expressions/expression_impl_factory.h"
#include "dali/operators/math/expressions/expression_tree.h"
#include "dali/operators/math/expressions/broadcasting.h"
#include "dali/pipeline/operator/op_spec.h"
#include "dali/pipeline/workspace/workspace.h"

namespace dali {
namespace expr {

template <ArithmeticOp op, typename Result, typename Input>
class ExprImplCpuT : public ExprImplBase {
 public:
  void Execute(ExprImplContext &ctx,
               span<const SampleDesc> samples,
               span<const TileDesc> tiles) override {
    assert(tiles.size() == 1 &&
           "CPU Expression implementation can handle only one tile at a time");
    const auto &tile = tiles[0];
    const auto &sample = samples[tile.sample_idx];
    auto output = static_cast<Result *>(sample.output.data);
    auto input = static_cast<const Input *>(sample.args[0].data);
    Execute(output, input, tile.offset, tile.size);
  }

 private:
  using meta_t = arithm_meta<op, CPUBackend>;

  static void Execute(Result *result, const Input *i0, int64_t offset, int64_t extent) {
    int64_t end = offset + extent;
    for (int64_t i = offset; i < end; i++) {
      result[i] = meta_t::impl(i0[i]);
    }
  }
};

template <ArithmeticOp op, typename Result, typename Left, typename Right>
class ExprImplCpuTT : public ExprImplBase {
 public:
  void Execute(ExprImplContext &ctx,
               span<const SampleDesc> samples,
               span<const TileDesc> tiles) override {
    assert(tiles.size() == 1 &&
           "CPU Expression implementation can handle only one tile at a time");
    const auto &tile = tiles[0];
    const auto &sample = samples[tile.sample_idx];
    auto &output = sample.output;
    auto *output_ptr = static_cast<Result *>(output.data);
    auto &left = sample.args[0];
    const auto *left_ptr = static_cast<const Left *>(sample.args[0].data);
    auto &right = sample.args[1];
    const auto *right_ptr = static_cast<const Right *>(sample.args[1].data);

    // Shapes are simplified so that we end up with 1D operands when both operands
    // have the same shape.
    if (sample.args[0].shape.sample_dim() == 1) {
      assert(sample.args[1].shape.sample_dim() == 1);
      Execute(output_ptr, left_ptr, right_ptr, tile.offset, tile.size);
    } else {
      assert(tile.offset == 0);
      assert(tile.size == volume(sample.output.shape));
      Execute(output_ptr, output.shape.data(), output.strides.data(),
              left_ptr, left.strides.data(), right_ptr, right.strides.data(),
              sample.output.shape.sample_dim());
    }
  }

 private:
  using meta_t = arithm_meta<op, CPUBackend>;

  static void Execute(Result *result, const Left *l, const Right *r,
                      int64_t offset, int64_t extent) {
    int64_t end = offset + extent;
    for (int64_t i = offset; i < end; i++) {
      result[i] = meta_t::impl(l[i], r[i]);
    }
  }

  template <int ndim>
  static void Execute(Result *out, const int64_t *out_shape, const int64_t *out_strides,
                      const Left* in0, const int64_t *in0_strides,
                      const Right* in1, const int64_t *in1_strides,
                      std::integral_constant<int, ndim>) {
    static_assert(ndim > 1);
    for (int64_t i = 0; i < *out_shape; i++) {
      Execute(out, out_shape + 1, out_strides + 1,
              in0, in0_strides + 1, in1, in1_strides + 1,
              std::integral_constant<int, ndim - 1>());
      in0 += *in0_strides;
      in1 += *in1_strides;
      out += *out_strides;
    }
  }

  static void Execute(Result *out, const int64_t *out_shape, const int64_t *out_strides,
                      const Left* in0, const int64_t *in0_strides,
                      const Right* in1, const int64_t *in1_strides,
                      std::integral_constant<int, 1>) {
    for (int64_t i = 0; i < *out_shape; i++) {
      *out = meta_t::impl(*in0, *in1);
      in0 += *in0_strides;
      in1 += *in1_strides;
      out += *out_strides;
    }
  }

  static void Execute(Result *out, const int64_t *out_shape, const int64_t *out_strides,
                      const Left* in0, const int64_t *in0_strides,
                      const Right* in1, const int64_t *in1_strides, int ndim) {
    VALUE_SWITCH(ndim, Dims, ARITHM_OPS_ALLOWED_DIMS, (
      return Execute(
        out, out_shape, out_strides,
        in0, in0_strides,
        in1, in1_strides,
        std::integral_constant<int, Dims>());
    ), assert(false););  // NOLINT
  }
};

template <ArithmeticOp op, typename Result, typename Left, typename Right>
class ExprImplCpuCT : public ExprImplBase {
 public:
  void Execute(ExprImplContext &ctx,
               span<const SampleDesc> samples,
               span<const TileDesc> tiles) override {
    assert(tiles.size() == 1 &&
           "CPU Expression implementation can handle only one tile at a time");
    const auto &tile = tiles[0];
    const auto &sample = samples[tile.sample_idx];
    auto output = static_cast<Result *>(sample.output.data);
    auto left_ptr = static_cast<const Left *>(sample.args[0].data);
    auto right_ptr = static_cast<const Right *>(sample.args[1].data);
    Execute(output, *left_ptr, right_ptr, tile.offset, tile.size);
  }

 private:
  using meta_t = arithm_meta<op, CPUBackend>;

  static void Execute(Result *result, Left l, const Right *r, int64_t offset, int64_t extent) {
    int64_t end = offset + extent;
    for (int64_t i = offset; i < end; i++) {
      result[i] = meta_t::impl(l, r[i]);
    }
  }
};

template <ArithmeticOp op, typename Result, typename Left, typename Right>
class ExprImplCpuTC : public ExprImplBase {
 public:
  void Execute(ExprImplContext &ctx,
               span<const SampleDesc> samples,
               span<const TileDesc> tiles) override {
    assert(tiles.size() == 1 &&
           "CPU Expression implementation can handle only one tile at a time");
    const auto &tile = tiles[0];
    const auto &sample = samples[tile.sample_idx];
    auto output = static_cast<Result *>(sample.output.data);
    auto left_ptr = static_cast<const Left *>(sample.args[0].data);
    auto right_ptr = static_cast<const Right *>(sample.args[1].data);
    Execute(output, left_ptr, *right_ptr, tile.offset, tile.size);
  }

 private:
  using meta_t = arithm_meta<op, CPUBackend>;

  static void Execute(Result *result, const Left *l, Right r, int64_t offset, int64_t extent) {
    int64_t end = offset + extent;
    for (int64_t i = offset; i < end; i++) {
      result[i] = meta_t::impl(l[i], r);
    }
  }
};

// Ternary operators

template <ArithmeticOp op, typename Result,
          bool IsFirstTensor, bool IsSecondTensor, bool IsThirdTensor>
class ExprImplCpuTernary : public ExprImplBase {
 public:
  void Execute(ExprImplContext &ctx,
               span<const SampleDesc> samples,
               span<const TileDesc> tiles) override {
    assert(tiles.size() == 1 &&
           "CPU Expression implementation can handle only one tile at a time");
    const auto &tile = tiles[0];
    const auto &sample = samples[tile.sample_idx];
    auto output = static_cast<Result *>(sample.output.data);

    auto &first = sample.args[0];
    auto &second = sample.args[1];
    auto &third = sample.args[2];

    if (sample.output.shape.sample_dim() == 1) {
      Execute(output,
              expression_detail::Pass<IsFirstTensor, Result>(first.data, first.dtype),
              first.dtype,
              expression_detail::Pass<IsSecondTensor, Result>(second.data, second.dtype),
              second.dtype,
              expression_detail::Pass<IsThirdTensor, Result>(third.data, third.dtype),
              third.dtype, tile.offset, tile.size);
    } else {
      assert(tile.offset == 0);
      assert(tile.size == volume(sample.output.shape));
      Execute(output, sample.output.shape.data(), sample.output.strides.data(),
              expression_detail::Pass<IsFirstTensor, Result>(first.data, first.dtype),
              first.dtype, first.strides.data(),
              expression_detail::Pass<IsSecondTensor, Result>(second.data, second.dtype),
              second.dtype, second.strides.data(),
              expression_detail::Pass<IsThirdTensor, Result>(third.data, third.dtype),
              third.dtype, third.strides.data(),
              sample.output.shape.sample_dim());
    }
  }

 private:
  using meta_t = arithm_meta<op, CPUBackend>;

  static void Execute(Result *result,
                      expression_detail::param_t<IsFirstTensor, Result> first,
                      DALIDataType first_type,
                      expression_detail::param_t<IsSecondTensor, Result> second,
                      DALIDataType second_type,
                      expression_detail::param_t<IsThirdTensor, Result> third,
                      DALIDataType third_type,
                      int64_t offset, int64_t extent) {
    int64_t end = offset + extent;
    for (int64_t i = offset; i < end; i++) {
      result[i] = meta_t::impl(expression_detail::Access<Result>(first, i, first_type),
                               expression_detail::Access<Result>(second, i, second_type),
                               expression_detail::Access<Result>(third, i, third_type));
    }
  }

  template <int ndim>
  static void Execute(Result *out, const int64_t *out_shape, const int64_t *out_strides,
                      expression_detail::param_t<IsFirstTensor, Result> first,
                      DALIDataType first_type, int64_t first_offset,
                      const int64_t *first_strides,
                      expression_detail::param_t<IsSecondTensor, Result> second,
                      DALIDataType second_type, int64_t second_offset,
                      const int64_t *second_strides,
                      expression_detail::param_t<IsThirdTensor, Result> third,
                      DALIDataType third_type, int64_t third_offset,
                      const int64_t *third_strides,
                      std::integral_constant<int, ndim>) {
    static_assert(ndim > 1);
    for (int64_t i = 0; i < *out_shape; i++) {
      Execute(out, out_shape + 1, out_strides + 1,
              first, first_type, first_offset, first_strides + 1,
              second, second_type, second_offset, second_strides + 1,
              third, third_type, third_offset, third_strides + 1,
              std::integral_constant<int, ndim - 1>());
      first_offset += *first_strides;
      second_offset += *second_strides;
      third_offset += *third_strides;
      out += *out_strides;
    }
  }

  static void Execute(Result *out, const int64_t *out_shape, const int64_t *out_strides,
                      expression_detail::param_t<IsFirstTensor, Result> first,
                      DALIDataType first_type, int64_t first_offset,
                      const int64_t *first_strides,
                      expression_detail::param_t<IsSecondTensor, Result> second,
                      DALIDataType second_type, int64_t second_offset,
                      const int64_t *second_strides,
                      expression_detail::param_t<IsThirdTensor, Result> third,
                      DALIDataType third_type, int64_t third_offset,
                      const int64_t *third_strides,
                      std::integral_constant<int, 1>) {
    for (int64_t i = 0; i < *out_shape; i++) {
      *out = meta_t::impl(expression_detail::Access<Result>(first, first_offset, first_type),
                          expression_detail::Access<Result>(second, second_offset, second_type),
                          expression_detail::Access<Result>(third, third_offset, third_type));
      first_offset += *first_strides;
      second_offset += *second_strides;
      third_offset += *third_strides;
      out += *out_strides;
    }
  }

  static void Execute(Result *out, const int64_t *out_shape, const int64_t *out_strides,
                      expression_detail::param_t<IsFirstTensor, Result> first,
                      DALIDataType first_type, const int64_t *first_strides,
                      expression_detail::param_t<IsSecondTensor, Result> second,
                      DALIDataType second_type, const int64_t *second_strides,
                      expression_detail::param_t<IsThirdTensor, Result> third,
                      DALIDataType third_type, const int64_t *third_strides,
                      int ndim) {
    VALUE_SWITCH(ndim, Dims, ARITHM_OPS_ALLOWED_DIMS, (
      return Execute(
        out, out_shape, out_strides,
        first, first_type, 0, first_strides,
        second, second_type, 0, second_strides,
        third, third_type, 0, third_strides,
        std::integral_constant<int, Dims>());
    ), assert(false););  // NOLINT
  }
};

}  // namespace expr
}  // namespace dali

#endif  // DALI_OPERATORS_MATH_EXPRESSIONS_EXPRESSION_IMPL_CPU_H_
