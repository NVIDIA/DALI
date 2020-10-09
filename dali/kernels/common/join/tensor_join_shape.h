// Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
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

#ifndef DALI_KERNELS_COMMON_JOIN_TENSOR_JOIN_SHAPE_H_
#define DALI_KERNELS_COMMON_JOIN_TENSOR_JOIN_SHAPE_H_

#include <cassert>
#include "dali/core/span.h"
#include "dali/core/tensor_shape.h"
#include "dali/core/tensor_shape_print.h"
#include "dali/core/format.h"
#include "dali/core/error_handling.h"

namespace dali {
namespace kernels {
namespace tensor_join {

constexpr int joined_ndim(int ndim, bool new_axis) {
  return ndim < 0 ? ndim : ndim + new_axis;
}

inline TensorShape<> ConcatenatedShape(span<const TensorShape<>> in_shapes, int axis) {
  TensorShape<> ret;
  auto &insh = in_shapes[0];
  ret.resize(insh.size() + 1);
  for (int i = 0; i < axis; i++) {
    ret[i] = insh[i];
  }
  ret[axis] = in_shapes.size();
  for (int i = axis + 1; i < ret.size(); i++) {
    ret[i] = insh[i - 1];
  }
  return ret;
}

inline TensorShape<> StackedShape(span<const TensorShape<>> in_shapes, int axis) {
  TensorShape<> ret = in_shapes[0];
  ret[axis] = 0;
  for (const auto &sh : in_shapes) {
    ret[axis] += sh[axis];
  }
  return ret;
}

inline TensorShape<> JoinedShape(span<const TensorShape<>> in_shapes, int axis, bool new_axis) {
  return new_axis ? ConcatenatedShape(in_shapes, axis) : StackedShape(in_shapes, axis);
}


template <typename GetInListShapeFunc>
void CheckJoinedShapes(GetInListShapeFunc &&in_shape, int njoin, int axis, bool new_axis) {
  if (njoin < 1)
    return;
  const auto &shape_0 = *in_shape(0);
  int N = shape_0.num_samples();
  int D = shape_0.sample_dim();
  DALI_ENFORCE(axis >= 0 && axis < D + new_axis, make_string(
    "Axis index ", axis, " is invalid for ", D, "-D tensor ",
    (new_axis ? "stacking" : "concatenation"), "."));

  for (int t = 1; t < njoin; t++) {
    const auto &shape_t = *in_shape(t);
    if (new_axis) {
      DALI_ENFORCE(shape_t == shape_0, make_string("Tensor stacking requires that all "
        "tensors being stacked are of equal shape. The first batch has a shape: ", shape_0,
        "\nbatch ", t, " has a shape ", shape_t));
    } else {
      for (int i = 0; i < N; i++) {
        for (int d = 0; d < D; d++) {
          if (d == axis)
            continue;
          auto extent = shape_t.tensor_shape_span(i)[d];
          auto ref_extent = shape_0.tensor_shape_span(i)[d];
          DALI_ENFORCE(extent == ref_extent, make_string("Concatenated tensors can differ in "
            "size only in the concatenation axis (", axis, ").\nTensor ", i, " in batch ", t,
            " has a shape ", shape_t[i], " which is not compatible with respective tensor in "
            "batch 0 of shape ", shape_0[i], ". Mismatch in axis ", d, "."));
        }
      }
    }
  }
}

template <int out_ndim, typename GetInListShapeFunc>
void JoinedShape(TensorListShape<out_ndim> &out,
                 GetInListShapeFunc &&in_shape, int njoin, int axis, bool new_axis) {
  if (njoin == 0) {
    out.resize(0, joined_ndim(in_shape(0)->sample_dim(), new_axis));
    return;
  }

  CheckJoinedShapes(in_shape, njoin, axis, new_axis);

  int N = in_shape(0)->num_samples();
  int D = in_shape(0)->sample_dim();
  out.resize(N, joined_ndim(in_shape(0)->sample_dim(), new_axis));

  int64_t in_volume = 0;
  for (int  t = 0; t < njoin; t++)
    in_volume += in_shape(t)->num_elements();

  for (int i = 0; i < N; i++) {
    auto out_ts = out.tensor_shape_span(i);

    // copy outer extents, up to `axis`
    int oa = 0, ia = 0;  // input axis, output axis
    for (; ia < axis; ia++, oa++)
      out_ts[oa] = in_shape(0)->tensor_shape_span(i)[ia];

    if (new_axis) {
      out_ts[oa++] = njoin;  // new axis - number of joined tensor
    } else {
      out_ts[oa] = 0;
      // join along existing axis - sum the extents
      for (int t = 0; t < njoin; t++) {
        out_ts[oa] += in_shape(t)->tensor_shape_span(i)[ia];
      }
      oa++, ia++;  // advance both input and output
    }

    // copy remaining inner extents
    for (; ia < D; ia++, oa++)
      out_ts[oa] = in_shape(0)->tensor_shape_span(i)[ia];
  }

  assert(out.num_elements() == in_volume);
}

}  // namespace tensor_join
}  // namespace kernels
}  // namespace dali

#endif  // DALI_KERNELS_COMMON_JOIN_TENSOR_JOIN_SHAPE_H_
