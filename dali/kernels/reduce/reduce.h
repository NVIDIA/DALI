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

#ifndef DALI_KERNELS_REDUCE_REDUCE_H_
#define DALI_KERNELS_REDUCE_REDUCE_H_

#include <cassert>
#include <vector>
#include "dali/kernels/kernel.h"
#include "dali/core/format.h"
#include "dali/core/small_vector.h"
#include "dali/core/span.h"
#include "dali/core/static_switch.h"
#include "dali/core/tensor_view.h"
#include "dali/core/util.h"

namespace dali {
namespace kernels {
namespace reductions {

struct identity {
  template <typename T>
  T &&operator()(T &&x) const noexcept {
    return std::forward<T>(x);
  }
};

struct square {
  template <typename T>
  auto operator()(const T &x) const noexcept {
    return x * x;
  }
};

template <typename Mean>
struct variance {
  Mean mean = 0;
  template <typename T>
  auto operator()(const T &x) const noexcept {
    auto d = x - mean;
    return d * d;
  }
};

struct sum {
  template <typename Acc, typename Addend>
  DALI_HOST_DEV DALI_FORCEINLINE
  void operator()(Acc &acc, const Addend &val) const noexcept {
    acc += val;
  }
};

}  // namespace reductions


constexpr int kTreeReduceThreshold = 32;

template <typename Dst, typename Src, typename Mapping, typename Reduction>
void reduce1D(Dst &reduced, const Src *data, int64_t stride, int64_t n,
              const Mapping &M, const Reduction &R) {
  if (n > kTreeReduceThreshold) {
    int64_t m = n >> 1;
    Dst tmp1 = 0, tmp2 = 0;
    // reduce first half and accumulate
    reduce1D(tmp1, data, stride, m, M, R);
    // reduce second half and accumulate
    reduce1D(tmp2, data + m * stride, stride, n - m, M, R);
    R(tmp1, tmp2);
    R(reduced, tmp1);
  } else {
    // reduce to a temporary
    Dst tmp = 0;
    for (int64_t i = 0; i < n; i++) {
      R(tmp, M(data[i * stride]));
    }
    // accumulate in target value
    R(reduced, tmp);
  }
}

template <typename Backend, typename T>
struct StridedTensor {
  T *data = nullptr;
  int dim() const noexcept { return size.size(); }
  SmallVector<int, 6> stride, size;
};


template <typename Dst, typename Src, typename Mapping, typename Reduction>
void reduce(Dst &reduced, const StridedTensor<StorageCPU, Src> &in,
            const Mapping &M, const Reduction &R,
            int axis, int64_t extent, int64_t offset) {
  int64_t stride = in.stride[axis];
  if (axis == in.dim() - 1) {
    Dst tmp = 0;
    reduce1D(tmp, in.data + offset, stride, in.size[axis], M, R);
    R(reduced, tmp);
  } else {
    int64_t sub_v = volume(in.size.begin() + axis + 1, in.size.end());
    if (!sub_v)
      sub_v = 1;
    if (extent >= 4 && extent * sub_v > kTreeReduceThreshold) {
      Dst tmp1 = 0, tmp2 = 0;
      int64_t mid = extent / 2;
      reduce(tmp1, in, M, R, axis, mid, offset);
      reduce(tmp2, in, M, R, axis, extent - mid, offset + mid * stride);
      R(tmp1, tmp2);
      R(reduced, tmp1);
    } else {
      for (int64_t i = 0; i < extent; i++) {
        Dst tmp = 0;
        reduce(tmp, in, M, R, axis + 1, in.size[axis + 1], offset + i * stride);
        R(reduced, tmp);
      }
    }
  }
}

template <typename Dst, typename Src, typename Mapping, typename Reduction>
void reduce(Dst &reduced, const StridedTensor<StorageCPU, Src> &in,
            const Mapping &M, const Reduction &R, int64_t offset) {
  reduce(reduced, in, M, R, 0, in.size[0], offset);
}

template <typename Dst, typename Src, typename Actual>
struct ReduceBase {

  void Setup(const OutTensorCPU<Dst, -1> &out,
             const InTensorCPU<Src, -1> &in,
             span<const int> axes) {
    if (in.shape.size() > 64)
      throw std::range_error("Reduce supports up to 64 dimensions");
    input = in;
    output = out;
    setup_axes(axes);
    setup_output();
    setup_input();
    assert(output.num_elements() == out.num_elements());
    This().PostSetup();
  }

  void PostSetup() {}

  void Run() {
    SmallVector<int64_t, 6> pos;
    pos.resize(output.dim());
    ReduceAxis(make_span(pos), 0, 0);
  }

  Actual &This() noexcept { return static_cast<Actual&>(*this); }
  const Actual &This() const noexcept { return static_cast<const Actual&>(*this); }

  reductions::identity GetMapping(span<int64_t> pos) const { return {}; }
  reductions::sum GetReduction() const { return {}; }
  Dst Potprocess(const Dst &x) const { return x; }

  void ReduceAxis(span<int64_t> pos, int a, int64_t offset = 0) {
    auto R = This().GetReduction();
    if (a == output.dim()) {
      Dst tmp = 0;
      reduce(tmp, strided_in, This().GetMapping(pos), R, offset);
      *output(pos) = This().Postprocess(tmp);
    } else {
      for (int64_t i = 0; i < output.shape[a]; i++) {
        pos[a] = i;
        ReduceAxis(pos, a+1, offset + i*step[a]);
      }
    }
  }

  void setup_axes(span<const int> _axes) {
    for (int axis : axes) {
      if (axis < 0 || axis >= ndim()) {
        throw std::range_error(make_string("Axis index out of range: ", axis, " not in 0..",
                               ndim()-1));
      }
    }

    SmallVector<int, 6> tmp_axes;
    tmp_axes = _axes;
    std::sort(tmp_axes.begin(), tmp_axes.end());
    int skipped = 0;
    int prev = -2;
    axes.clear();
    skipped_axes.clear();
    for (int i = 0; i < static_cast<int>(tmp_axes.size()); i++) {
      int axis = tmp_axes[i] - skipped;
      if (axis != prev + 1) {
        axes.push_back(axis);
      } else {
        assert(prev >= 0);
        input = collapse_dim(input, prev);
        skipped++;
        skipped_axes.push_back(tmp_axes[i]);
      }
      prev = axis;
    }
    axis_mask = 0;
    for (int axis : axes) {
      axis_mask |= static_cast<uint64_t>(1) << axis;
    }
  }

  void setup_output() {
    output.shape.resize(ndim() - axes.size());
    // full reduction?
    if (output.shape.empty())
      output.shape = { 1 };
  }

  void setup_input() {
    step.clear();
    SmallVector<int, 6> strides;
    strides.resize(ndim());
    int64_t v = 1;
    for (int i = ndim()-1; i >= 0; i--) {
      strides[i] = v;
      v *= input.shape[i];
    }

    strided_in = {};
    strided_in.data = input.data;
    for (int i = 0, oaxis = 0; i < ndim(); i++) {
      if (is_reduced_axis(i)) {
        strided_in.stride.push_back(strides[i]);
        strided_in.size.push_back(input.shape[i]);
      } else {
        step.push_back(strides[i]);
        output.shape[oaxis] = input.shape[i];
        oaxis++;
      }
    }
  }

  DALI_FORCEINLINE int ndim() const noexcept { return input.shape.size(); }
  DALI_FORCEINLINE bool is_reduced_axis(int axis) const noexcept {
    return axis_mask & (static_cast<uint64_t>(1) << axis);
  }

  OutTensorCPU<Dst, -1> output;
  InTensorCPU<Src, -1> input;
  SmallVector<int, 6> axes, skipped_axes;
  StridedTensor<StorageCPU, const Src> strided_in;
  SmallVector<int64_t, 6> step;
  uint64_t axis_mask = 0;
};


template <typename Dst, typename Src>
struct Mean : ReduceBase<Dst, Src, Mean<Dst, Src>> {
  void PostSetup() {
    norm_factor = 1;
    for (auto a : this->axes)
      norm_factor *= this->input.shape[a];
  }

  Dst Postprocess(Dst x) const {
    return x / norm_factor;
  }

  std::conditional_t<std::is_same<Dst, double>::value, double, float> norm_factor = 1;
};


template <typename Dst, typename Src, typename MeanType = Dst>
struct StdDev : ReduceBase<Dst, Src, StdDev<Dst, Src, MeanType>> {
  using Base = ReduceBase<Dst, Src, StdDev<Dst, Src, MeanType>>;
  InTensorCPU<MeanType, -1> mean;

  void Setup(const OutTensorCPU<Dst, -1> &out,
             const InTensorCPU<Dst, -1> &in,
             span<const int> axes,
             const InTensorCPU<Dst, -1> &mean) {
    assert(mean.shape == out.shape);
    Base::Setup(out, in, axes);
    this->mean = mean;
    this->mean.shape = this->output.shape;
  }

  void PostSetup() {
    norm_factor = 1;
    for (auto a : this->axes)
      norm_factor /= this->input.shape[a];
  }

  reductions::variance<MeanType> GetMapping(span<int64_t> pos) const {
    return { *mean(pos) };
  }

  Dst Postprocess(Dst x) const {
    return std::sqrt(x * norm_factor);
  }

  std::conditional_t<std::is_same<Dst, double>::value, double, float> norm_factor = 1;
};


}  // namespace kernels
}  // namespace dali

#endif  // DALI_KERNELS_REDUCE_REDUCE_H_
