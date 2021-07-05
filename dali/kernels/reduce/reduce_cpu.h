// Copyright (c) 2020-2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef DALI_KERNELS_REDUCE_REDUCE_CPU_H_
#define DALI_KERNELS_REDUCE_REDUCE_CPU_H_

#include <cassert>
#include <utility>
#include <vector>
#include "dali/kernels/kernel.h"
#include "dali/kernels/common/utils.h"
#include "dali/kernels/reduce/reductions.h"
#include "dali/core/format.h"
#include "dali/core/small_vector.h"
#include "dali/core/span.h"
#include "dali/core/static_switch.h"
#include "dali/core/tensor_shape_print.h"
#include "dali/core/tensor_view.h"
#include "dali/core/util.h"

namespace dali {
namespace kernels {

namespace reduce_impl {

constexpr int kTreeReduceThreshold = 32;

template <int static_stride, typename Dst, typename Src, typename Preprocessor, typename Reduction>
void reduce1D_stride(Dst &reduced, const Src *data, int64_t dynamic_stride, int64_t n,
                     const Preprocessor &P, const Reduction &R) {
  const int64_t stride = static_stride < 0 ? dynamic_stride : static_stride;
  const Dst neutral = R.template neutral<Dst>();
  if (n > kTreeReduceThreshold) {
    int64_t m = n >> 1;
    Dst tmp1 = neutral, tmp2 = neutral;
    // reduce first half and accumulate
    reduce1D_stride<static_stride>(tmp1, data, stride, m, P, R);
    // reduce second half and accumulate
    reduce1D_stride<static_stride>(tmp2, data + m * stride, stride, n - m, P, R);
    R(tmp1, tmp2);
    R(reduced, tmp1);
  } else {
    // reduce to a temporary
    Dst tmp = neutral;
    for (int64_t i = 0; i < n; i++)
       R(tmp, P(data[i * stride]));
    // accumulate in target value
    R(reduced, tmp);
  }
}

template <typename Dst, typename Src, typename Preprocessor, typename Reduction>
void reduce1D(Dst &reduced, const Src *data, int64_t stride, int64_t n,
              const Preprocessor &P, const Reduction &R) {
  VALUE_SWITCH(stride, static_stride, (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 16),
    (reduce1D_stride<static_stride>(reduced, data, static_stride, n, P, R);),
    (reduce1D_stride<-1>(reduced, data, stride, n, P, R);)
  );  // NOLINT
}

template <typename Backend, typename T>
struct StridedTensor {
  T *data = nullptr;
  int dim() const noexcept { return size.size(); }
  SmallVector<int, 6> stride, size;
};

/// @brief Reduces a strided tensor slice to a scalar
template <typename Dst, typename Src, typename Preprocessor, typename Reduction>
void reduce(Dst &reduced, const StridedTensor<StorageCPU, Src> &in,
            const Preprocessor &P, const Reduction &R,
            int axis, int64_t extent, int64_t offset) {
  int64_t stride = in.stride[axis];
  const Dst neutral = R.template neutral<Dst>();
  if (axis == in.dim() - 1) {
    Dst tmp = neutral;
    reduce1D(tmp, in.data + offset, stride, in.size[axis], P, R);
    R(reduced, tmp);
  } else {
    int64_t sub_v = volume(in.size.begin() + axis + 1, in.size.end());
    if (extent >= 2 && extent * sub_v > kTreeReduceThreshold) {
      Dst tmp1 = neutral, tmp2 = neutral;
      int64_t mid = extent / 2;
      reduce(tmp1, in, P, R, axis, mid, offset);
      reduce(tmp2, in, P, R, axis, extent - mid, offset + mid * stride);
      R(tmp1, tmp2);
      R(reduced, tmp1);
    } else {
      for (int64_t i = 0; i < extent; i++) {
        Dst tmp = neutral;
        reduce(tmp, in, P, R, axis + 1, in.size[axis + 1], offset + i * stride);
        R(reduced, tmp);
      }
    }
  }
}

/// @brief Reduces a strided tensor to a scalar
template <typename Dst, typename Src, typename Preprocessor, typename Reduction>
void reduce(Dst &reduced, const StridedTensor<StorageCPU, Src> &in,
            const Preprocessor &P, const Reduction &R, int64_t offset) {
  reduce(reduced, in, P, R, 0, in.size[0], offset);
}

}  // namespace reduce_impl

/**
 * Base CRTP class for reduction. The pairwise reduction functor and Preprocessor functor
 * come from the Actual subclass.
 * As with any CRTP-based system, any non-private method can be shadowed by the Actual class.
 *
 * The default implementation uses pairwise reduction which is suitable for summation. If this kind
 * of reduction is not suitable, the Actual class should replace the whole Run method.
 *
 * The default postprocessing is an elementwise call to Actual::Postprocess. If different kind of
 * postprocessing is required, Actual should replace the PostprocessAll method instead.
 *
 * @tparam Actual - provides `GetPreprocessor`, `GetReduction`, `PostSetup` and `Postprocess`
 *         functions
 *         `GetPreprocessor` returns a unary functor that transforms an input value.
 *         `GetReduction` returns a binary functor that combines an accumulator with a new value
 *         `PostSetup` runs after default setup stage.
 *         `Postprocess` transforms the output value in some way (e.g. applies some normalization
 *         or weighting).
 *
 */
template <typename Dst, typename Src, typename Actual>
struct ReduceBaseCPU {
  void Setup(const OutTensorCPU<Dst, -1> &out,
             const InTensorCPU<Src, -1> &in,
             span<const int> axes) {
    if (in.shape.size() > 64)
      throw std::range_error("Reduce supports up to 64 dimensions");
    input = in;
    output = out;
    Actual &self = This();
    self.InitAxes(axes);
    self.CheckOutput();
    self.CollapseAxes();
    self.SetupOutput();
    self.SetupInput();
    assert(output.num_elements() == out.num_elements());
    self.PostSetup();
  }

  KernelRequirements Setup(
      KernelContext ctx,
      const OutTensorCPU<Dst, -1> &out,
      const InTensorCPU<Src, -1> &in,
      span<const int> axes) {
    Setup(out, in, axes);
    return KernelRequirements();
  }

  void PostSetup() {}

  void Run(bool clear = true, bool postprocess = true) {
    SmallVector<int64_t, 6> pos;
    pos.resize(output.dim());
    if (axes.empty()) {
      ReduceForEmptyAxes(make_span(pos));
    } else {
      ReduceAxis(clear, make_span(pos), 0, 0);
    }
    if (postprocess)
      This().PostprocessAll();
  }

  void Run(KernelContext ctx, bool clear = true, bool postprocess = true) {
    Run(clear, postprocess);
  }

  void PostprocessAll() {
    if (reinterpret_cast<decltype(&ReduceBaseCPU::Postprocess)>(&Actual::Postprocess) ==
        &ReduceBaseCPU::Postprocess)
      return;  // trivial postprocessing, nothing to do

    int64_t v = output.num_elements();
    for (int64_t i = 0; i < v; i++)
      output.data[i] = This().Postprocess(output.data[i]);
  }

  Actual &This() noexcept { return static_cast<Actual&>(*this); }
  const Actual &This() const noexcept { return static_cast<const Actual&>(*this); }

  /**
   * @brief Returns a unary function transforming the input values prior to reduction.
   *
   * When shadowed in the Actual class, the return type may differ.
   *
   * @param pos coordinates of the reduced value in the output tensor
   */
  identity GetPreprocessor(span<int64_t> pos) const { return {}; }

  /// @brief Returns a reduction functor, by default, a sum. Can be shadowed by Actual class.
  reductions::sum GetReduction() const { return {}; }

  /// @brief Transforms an output value. Can be shadowed by Actual class.
  Dst Postprocess(const Dst &x) const { return x; }

 protected:
  void ReduceAxis(bool clear, span<int64_t> pos, int axis, int64_t offset = 0) {
    auto R = This().GetReduction();
    if (axis == output.dim()) {
      Dst &r = *output(pos);
      if (clear) {
        r = R.template neutral<Dst>();
      }
      reduce_impl::reduce(r, strided_in, This().GetPreprocessor(pos), R, offset);
    } else {
      for (int64_t i = 0; i < output.shape[axis]; i++) {
        pos[axis] = i;
        ReduceAxis(clear, pos, axis+1, offset + i*step[axis]);
      }
    }
  }

  void ReduceForEmptyAxes(span<int64_t> pos) {
    auto P = This().GetPreprocessor(pos);
    for (int64_t i = 0; i < output.num_elements(); i++) {
      Dst preprocessed = P(input.data[i]);
      output.data[i] = preprocessed;
    }
  }

  void InitAxes(span<const int> _axes) {
    for (int axis : _axes) {
      if (axis < 0 || axis >= ndim()) {
        throw std::range_error(make_string("Axis index out of range: ", axis, " not in 0..",
                               ndim()-1));
      }
    }

    axes = _axes;
    axis_mask = 0;
    for (int axis : axes) {
      axis_mask |= 1_u64 << axis;
    }
  }

  void CheckOutput() {
    if (axis_mask == (1_u64 << ndim()) - 1) {
      DALI_ENFORCE(
        (output.dim() == 1 || output.dim() == 0 || output.dim() == input.dim()) &&
          output.num_elements() == 1,
        make_string("Full reduction produces a single value (possibly keeping reduced dimensions)."
        "\nOutput shape provided: ", output.shape));
    } else {
      TensorShape<> expected1, expected2 = input.shape;
      for (int i = 0; i < input.dim(); i++) {
        if (is_reduced_axis(i))
          expected2[i] = 1;
        else
          expected1.shape.push_back(input.shape[i]);
      }
      DALI_ENFORCE(output.shape == expected1 || output.shape == expected2, make_string(
        "Unexpected shape for reduction. Should be:  ", expected1, "  or  ", expected2, ",\ngot:  ",
        output.shape));
    }
  }


  void CollapseAxes() {
    SmallVector<int, 6> tmp_axes = std::move(axes);
    std::sort(tmp_axes.begin(), tmp_axes.end());
    int skipped = 0;
    int prev = -2;
    axes.clear();
    for (int i = 0; i < static_cast<int>(tmp_axes.size()); i++) {
      int axis = tmp_axes[i] - skipped;
      if (axis != prev + 1) {
        axes.push_back(axis);
        prev = axis;
      } else {
        assert(prev >= 0);
        input = collapse_dim(input, prev);
        skipped++;
      }
    }
    axis_mask = 0;
    for (int axis : axes) {
      axis_mask |= 1_u64 << axis;
    }
  }

  void SetupOutput() {
    output.shape.resize(ndim() - axes.size());
    // full reduction?
    if (output.shape.empty())
      output.shape = { 1 };
  }

  void SetupInput() {
    step.clear();
    auto strides = GetStrides(input.shape);

    strided_in = {};
    strided_in.data = input.data;
    int oaxis = 0;
    for (int i = 0; i < ndim(); i++) {
      if (is_reduced_axis(i)) {
        strided_in.stride.push_back(strides[i]);
        strided_in.size.push_back(input.shape[i]);
      } else {
        step.push_back(strides[i]);
        output.shape[oaxis] = input.shape[i];
        oaxis++;
      }
    }
    assert((oaxis == 0 && output.dim() == 1) || oaxis == output.dim());
  }

  DALI_FORCEINLINE int ndim() const noexcept { return input.shape.size(); }
  DALI_FORCEINLINE bool is_reduced_axis(int axis) const noexcept {
    return axis_mask & (1_u64 << axis);
  }

  OutTensorCPU<Dst, -1> output;
  InTensorCPU<Src, -1> input;
  SmallVector<int, 6> axes;
  reduce_impl::StridedTensor<StorageCPU, const Src> strided_in;
  SmallVector<int64_t, 6> step;
  uint64_t axis_mask = 0;
};

template <typename Dst, typename Src>
struct SumCPU : ReduceBaseCPU<Dst, Src, SumCPU<Dst, Src>> {
};


template <typename Dst, typename Src>
struct MinCPU : ReduceBaseCPU<Dst, Src, MinCPU<Dst, Src>> {
  reductions::min GetReduction() const { return {}; }
};


template <typename Dst, typename Src>
struct MaxCPU : ReduceBaseCPU<Dst, Src, MaxCPU<Dst, Src>> {
  reductions::max GetReduction() const { return {}; }
};


template <typename Dst, typename Src>
struct MeanCPU : ReduceBaseCPU<Dst, Src, MeanCPU<Dst, Src>> {
  void PostSetup() {
    int64_t v = 1;
    for (auto a : this->axes)
      v *= this->input.shape[a];
    norm_factor = 1.0 / v;
  }

  Dst Postprocess(Dst x) const {
    return x * norm_factor;
  }

  std::conditional_t<std::is_same<Dst, double>::value, double, float> norm_factor = 1;
};


template <typename Dst, typename Src>
struct MeanSquareCPU : ReduceBaseCPU<Dst, Src, MeanSquareCPU<Dst, Src>> {
  using Base = ReduceBaseCPU<Dst, Src, MeanSquareCPU<Dst, Src>>;

  void PostSetup() {
    int64_t v = 1;
    for (auto a : this->axes)
      v *= this->input.shape[a];
    norm_factor = 1.0 / v;
  }

  reductions::square GetPreprocessor(span<int64_t> pos) const { return {}; }

  Dst Postprocess(Dst x) const {
    return x * norm_factor;
  }

  std::conditional_t<std::is_same<Dst, double>::value, double, float> norm_factor = 1;
};


template <typename Dst, typename Src>
struct RootMeanSquareCPU : ReduceBaseCPU<Dst, Src, RootMeanSquareCPU<Dst, Src>> {
  using Base = ReduceBaseCPU<Dst, Src, RootMeanSquareCPU<Dst, Src>>;

  void PostSetup() {
    int64_t v = 1;
    for (auto a : this->axes)
      v *= this->input.shape[a];
    norm_factor = 1.0 / v;
  }

  reductions::square GetPreprocessor(span<int64_t> pos) const { return {}; }

  Dst Postprocess(Dst x) const {
    return std::sqrt(x * norm_factor);
  }

  std::conditional_t<std::is_same<Dst, double>::value, double, float> norm_factor = 1;
};

template <typename Dst, typename Src, typename MeanType = Dst>
struct VarianceCPU : ReduceBaseCPU<Dst, Src, VarianceCPU<Dst, Src, MeanType>> {
  using Base = ReduceBaseCPU<Dst, Src, VarianceCPU<Dst, Src, MeanType>>;
  InTensorCPU<MeanType, -1> mean;

  void Setup(
      const OutTensorCPU<Dst, -1> &out,
      const InTensorCPU<Src, -1> &in,
      span<const int> axes,
      const InTensorCPU<MeanType, -1> &mean) {
    assert(mean.shape == out.shape);
    Base::Setup(out, in, axes);
    this->mean = mean;
    this->mean.shape = this->output.shape;
  }

  KernelRequirements Setup(
      KernelContext ctx,
      const OutTensorCPU<Dst, -1> &out,
      const InTensorCPU<Src, -1> &in,
      span<const int> axes,
      const InTensorCPU<MeanType, -1> &mean,
      int ddof) {
    ddof_ = ddof;
    Setup(out, in, axes, mean);
    return KernelRequirements();
  }

  void PostSetup() {
    int64_t v = 1;
    for (auto a : this->axes)
      v *= this->input.shape[a];
    v -= ddof_;
    norm_factor = 1.0 / v;
  }

  reductions::variance<MeanType> GetPreprocessor(span<int64_t> pos) const {
    return { *mean(pos) };
  }

  Dst Postprocess(Dst x) const {
    return x * norm_factor;
  }

  std::conditional_t<std::is_same<Dst, double>::value, double, float> norm_factor = 1;
  int ddof_ = 0;
};

template <typename Dst, typename Src, typename MeanType = Dst>
struct StdDevCPU : ReduceBaseCPU<Dst, Src, StdDevCPU<Dst, Src, MeanType>> {
  using Base = ReduceBaseCPU<Dst, Src, StdDevCPU<Dst, Src, MeanType>>;
  InTensorCPU<MeanType, -1> mean;

  void Setup(
      const OutTensorCPU<Dst, -1> &out,
      const InTensorCPU<Src, -1> &in,
      span<const int> axes,
      const InTensorCPU<MeanType, -1> &mean) {
    assert(mean.shape == out.shape);
    Base::Setup(out, in, axes);
    this->mean = mean;
    this->mean.shape = this->output.shape;
  }

  KernelRequirements Setup(
      KernelContext ctx,
      const OutTensorCPU<Dst, -1> &out,
      const InTensorCPU<Src, -1> &in,
      span<const int> axes,
      const InTensorCPU<MeanType, -1> &mean,
      int ddof) {
    ddof_ = ddof;
    Setup(out, in, axes, mean);
    return KernelRequirements();
  }

  void PostSetup() {
    int64_t v = 1;
    for (auto a : this->axes)
      v *= this->input.shape[a];
    v -= ddof_;
    norm_factor = 1.0 / v;
  }

  reductions::variance<MeanType> GetPreprocessor(span<int64_t> pos) const {
    return { *mean(pos) };
  }

  Dst Postprocess(Dst x) const {
    return std::sqrt(x * norm_factor);
  }

  std::conditional_t<std::is_same<Dst, double>::value, double, float> norm_factor = 1;
  int ddof_ = 0;
};


}  // namespace kernels
}  // namespace dali

#endif  // DALI_KERNELS_REDUCE_REDUCE_CPU_H_
