// Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "dali/operators/math/expressions/broadcasting.h"
#include "dali/operators/math/expressions/arithmetic_meta.h"

namespace dali {

void CheckNumSamples(span<const TensorListShape<>> shapes) {
  DALI_ENFORCE(shapes.size() >= 1);
  int nsamples = shapes[0].num_samples();
  for (int i = 1; i < shapes.size(); i++) {
    DALI_ENFORCE(nsamples == shapes[i].num_samples());
  }
}

template <typename Shape>
int BroadcastNdimImpl(span<Shape> shapes) {
  DALI_ENFORCE(shapes.size() >= 1);
  int ndim = shapes[0].size();
  for (int i = 1; i < shapes.size(); i++) {
    ndim = std::max(ndim, shapes[i].size());
  }
  return ndim;
}

int BroadcastNdim(span<const TensorShape<>> shapes) {
  return BroadcastNdimImpl(shapes);
}

int BroadcastNdim(span<const TensorListShape<>> shapes) {
  DALI_ENFORCE(shapes.size() >= 1);
  int ndim = shapes[0].sample_dim();
  for (int i = 1; i < shapes.size(); i++) {
    ndim = std::max(ndim, shapes[i].sample_dim());
  }
  return ndim;
}

template <typename Shape>
void BroadcastShapeImpl(Shape& result, span<const Shape> shapes) {
  DALI_ENFORCE(shapes.size() >= 1);
  if (shapes.size() == 1) {
    result = shapes[0];
    return;
  }
  int ndim = BroadcastNdim(shapes);
  DALI_ENFORCE(result.size() == ndim);

  for (int d = 0; d < ndim; d++) {
    auto &res = result[result.size() - 1 - d];
    res = 1;
    for (int i = 0; i < shapes.size(); i++) {
      auto &sh = shapes[i];
      auto extent = d < sh.size() ? sh[sh.size() - 1 - d] : 1;
      if (extent > 1 && extent != res) {
        if (res > 1)
          DALI_FAIL(make_string("Can't broadcast extents ", extent, " and ", res));
        res = extent;
      }
    }
  }
}

void BroadcastShape(TensorShape<>& result, span<const TensorShape<>> shapes) {
  int ndim = BroadcastNdim(shapes);
  result.resize(ndim);
  BroadcastShapeImpl(result, shapes);
}

void BroadcastShape(TensorListShape<>& result, span<const TensorListShape<>> shapes) {
  DALI_ENFORCE(shapes.size() >= 1);
  if (shapes.size() == 1) {
    result = shapes[0];
    return;
  }
  int ndim = BroadcastNdim(shapes);
  CheckNumSamples(shapes);
  int nsamples = shapes[0].num_samples();
  result.resize(nsamples, ndim);

  for (int sample_idx = 0; sample_idx < nsamples; sample_idx++) {
    auto res_sample = result.tensor_shape_span(sample_idx);
    for (int d = 0; d < ndim; d++) {
      auto &res = res_sample[res_sample.size() - 1 - d];
      res = 1;
      for (int i = 0; i < shapes.size(); i++) {
        auto sh = shapes[i].tensor_shape_span(sample_idx);
        auto extent = d < sh.size() ? sh[sh.size() - 1 - d] : 1;
        if (extent > 1 && extent != res) {
          if (res > 1)
            DALI_FAIL(make_string("Can't broadcast extents ", extent, " and ", res));
          res = extent;
        }
      }
    }
  }
}

template <typename Shape>
bool CanBroadcastImpl(span<Shape> shapes) {
  if (shapes.size() < 2) {
    return true;
  }
  int ndim = BroadcastNdim(shapes);
  for (int d = 0; d < ndim; d++) {
    int64_t val = 1;
    for (int i = 0; i < shapes.size(); i++) {
      auto &sh = shapes[i];
      auto extent = d < sh.size() ? sh[sh.size() - 1 - d] : 1;
      if (extent > 1 && extent != val) {
        if (val > 1)
          return false;
        val = extent;
      }
    }
  }
  return true;
}

bool CanBroadcast(span<const TensorShape<>> shapes) {
  return CanBroadcastImpl(shapes);
}

bool CanBroadcast(span<const TensorListShape<>> shapes) {
  if (shapes.size() < 2) {
    return true;
  }
  int ndim = BroadcastNdim(shapes);
  int nsamples = shapes[0].num_samples();
  for (int sample_idx = 0; sample_idx < nsamples; sample_idx++) {
    for (int d = 0; d < ndim; d++) {
      int64_t val = 1;
      for (int i = 0; i < shapes.size(); i++) {
        const auto &sh = shapes[i].tensor_shape_span(sample_idx);
        auto extent = d < sh.size() ? sh[sh.size() - 1 - d] : 1;
        if (extent > 1 && extent != val) {
          if (val > 1)
            return false;
          val = extent;
        }
      }
    }
  }
  return true;
}

template <typename Shape>
bool NeedBroadcastImpl(span<Shape> shapes) {
  if (shapes.size() < 2)
    return false;
  auto *prev_sh = &shapes[0];
  for (int i = 1; i < shapes.size(); i++) {
    if (IsScalarLike(*prev_sh)) {
      prev_sh = &shapes[i];
    } else if (!IsScalarLike(shapes[i]) && *prev_sh != shapes[i]) {
      return true;
    }
  }
  return false;
}

bool NeedBroadcasting(span<const TensorShape<>> shapes) {
  return NeedBroadcastImpl(shapes);
}

bool NeedBroadcasting(span<const TensorListShape<>> shapes) {
  return NeedBroadcastImpl(shapes);
}

TensorShape<> StridesForBroadcasting(const TensorShape<> &out_sh, const TensorShape<> &in_sh,
                                     const TensorShape<> &in_strides) {
  TensorShape<> strides;
  assert(in_sh.size() == in_strides.size());
  assert(in_sh.size() <= out_sh.size());
  int out_ndim = out_sh.size();
  strides.shape.resize(out_ndim, 0);
  for (int i = (out_ndim - in_sh.size()); i < out_ndim; i++) {
    assert(in_sh[i] == out_sh[i] || in_sh[i] == 1);
    if (in_sh[i] == out_sh[i])
      strides[i] = in_strides[i];
    else
      strides[i] = 0;
  }
  return strides;
}

void ExpandToNDims(TensorShape<> &sh, int ndim) {
  assert(sh.size() <= ndim);
  if (sh.size() == ndim)
    return;
  TensorShape<> sh2;
  sh2.shape.resize(ndim);
  int i = 0;
  for (; i < (ndim - sh.size()); i++)
    sh2[i] = 1;
  for (int j = 0; j < sh.size(); j++, i++)
    sh2[i] = sh[j];
  std::swap(sh, sh2);
}

void SimplifyShapesForBroadcasting(TensorShape<>& lhs, TensorShape<> &rhs) {
  // First, if needed expand dimensions
  int full_ndim = std::max(lhs.size(), rhs.size());
  if (lhs.size() != rhs.size()) {
    ExpandToNDims(lhs, full_ndim);
    ExpandToNDims(rhs, full_ndim);
  }

  int i = 0;
  SmallVector<std::pair<int, int>, 5> group_dims;
  while (i < full_ndim) {
    if (lhs[i] != rhs[i]) {
      i++;
      continue;
    }
    int j = i;
    for (; j < full_ndim; j++) {
      if (lhs[j] != rhs[j]) break;
    }
    if (i < j) {
      group_dims.emplace_back(i, j - i);
    }
    i = j;
  }

  lhs = collapse_dims(lhs, group_dims);
  rhs = collapse_dims(rhs, group_dims);
}

}  // namespace dali
