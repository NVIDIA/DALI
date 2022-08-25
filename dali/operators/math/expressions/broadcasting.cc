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
#include <string>
#include <vector>
#include "dali/operators/math/expressions/arithmetic_meta.h"

namespace dali {

void CheckNumSamples(span<const TensorListShape<>*> shapes) {
  if (shapes.size() <= 1)
    return;
  int nsamples = shapes[0]->num_samples();
  for (int i = 1; i < shapes.size(); i++) {
    DALI_ENFORCE(nsamples == shapes[i]->num_samples(),
                 make_string("Number of samples should match. Got shapes with ", nsamples, " and ",
                             shapes[i]->num_samples(), " respectively."));
  }
}

template <typename Shape>
int BroadcastNdimImpl(span<const Shape*> shapes) {
  DALI_ENFORCE(shapes.size() >= 1);
  int ndim = shapes[0]->sample_dim();
  for (int i = 1; i < shapes.size(); i++) {
    ndim = std::max(ndim, shapes[i]->sample_dim());
  }
  return ndim;
}

int BroadcastNdim(span<const TensorShape<>*> shapes) {
  return BroadcastNdimImpl(shapes);
}

int BroadcastNdim(span<const TensorListShape<>*> shapes) {
  return BroadcastNdimImpl(shapes);
}

/**
 * @brief Implementation helper for `BroadcastShape` and `CanBroadcast`
 *        The function updates val with the d-th dimension of `sh`, or returns false.
 *        `val` can be updated with a new value if its current value is 1 or equal to the new one.
 * @tparam ShapeLike e.g. span<const int64_t>, TensorShape<>
 * @param val extent value to update (in an output shape)
 * @param sh one of input shape
 * @param d one of the dimension indices
 * @return true if val could be updated (broadcastable)
 * @return false if val could not be updated (non broadcastable)
 */
template <typename ShapeLike>
bool BroadcastShapeImpl(int64_t& val, const ShapeLike& sh, int d) {
  int ndim = sh.size();
  auto extent = d < ndim ? sh[ndim - 1 - d] : 1;
  if (extent > 1 && extent != val) {
    if (val > 1)
      return false;
    val = extent;
  }
  return true;
}

std::string PrintShape(const TensorShape<>& shape, int sample_idx, int d) {
  (void) sample_idx;
  return make_string(to_string(shape), "(d=", shape.sample_dim() - d - 1, ")");
}

std::string PrintShape(const TensorListShape<>& shape, int sample_idx, int d) {
  return make_string(to_string(shape[sample_idx]), " (d=", shape.sample_dim() - d - 1,
                     ", belonging to sample_idx=", sample_idx, ")");
}

template <typename Shapes>
std::string BroadcastErrorMessage(const Shapes& shapes, int64_t res,
                                  int sample_idx, int d) {
  std::stringstream ss;
  ss << "Can't broadcast " << d << "-th dimension of shapes: ";
  for (int j = 0; j < shapes.size(); j++) {
    ss << "\n" << PrintShape(*shapes[j], sample_idx, d);
    if (j == shapes.size() - 1)
      ss << "\n";
  }
  return ss.str();
}

void BroadcastShape(TensorShape<>& result, span<const TensorShape<>*> shapes) {
  DALI_ENFORCE(shapes.size() >= 1);
  if (shapes.size() == 1) {
    result = *shapes[0];
    return;
  }

  int ndim = BroadcastNdim(shapes);
  result.resize(ndim);

  for (int d = 0; d < ndim; d++) {
    auto &res = result[ndim - 1 - d];
    res = 1;
    for (int i = 0; i < shapes.size(); i++) {
      if (!BroadcastShapeImpl(res, *shapes[i], d)) {
        DALI_FAIL(BroadcastErrorMessage(shapes, res, i, d));
      }
    }
  }
}

void BroadcastShape(TensorListShape<>& result, span<const TensorListShape<>*> shapes) {
  DALI_ENFORCE(shapes.size() >= 1);
  if (shapes.size() == 1) {
    result = *shapes[0];
    return;
  }
  int ndim = BroadcastNdim(shapes);
  CheckNumSamples(shapes);
  int nsamples = shapes[0]->num_samples();
  result.resize(nsamples, ndim);

  for (int sample_idx = 0; sample_idx < nsamples; sample_idx++) {
    auto res_sample = result.tensor_shape_span(sample_idx);
    for (int d = 0; d < ndim; d++) {
      auto &res = res_sample[res_sample.size() - 1 - d];
      res = 1;
      for (int i = 0; i < shapes.size(); i++) {
        if (!BroadcastShapeImpl(res, shapes[i]->tensor_shape_span(sample_idx), d)) {
          DALI_FAIL(BroadcastErrorMessage(shapes, res, sample_idx, d));
        }
      }
    }
  }
}

bool CanBroadcast(span<const TensorShape<>*> shapes) {
  if (shapes.size() < 2) {
    return true;
  }
  int ndim = BroadcastNdim(shapes);
  for (int d = 0; d < ndim; d++) {
    int64_t val = 1;
    for (int i = 0; i < shapes.size(); i++) {
      if (!BroadcastShapeImpl(val, *shapes[i], d))
        return false;
    }
  }
  return true;
}

bool CanBroadcast(span<const TensorListShape<>*> shapes) {
  if (shapes.size() < 2) {
    return true;
  }
  int ndim = BroadcastNdim(shapes);
  int nsamples = shapes[0]->num_samples();
  for (int sample_idx = 0; sample_idx < nsamples; sample_idx++) {
    for (int d = 0; d < ndim; d++) {
      int64_t val = 1;
      for (int i = 0; i < shapes.size(); i++) {
        if (!BroadcastShapeImpl(val, shapes[i]->tensor_shape_span(sample_idx), d))
          return false;
      }
    }
  }
  return true;
}

template <typename Shape>
bool NeedBroadcastImpl(span<const Shape*> shapes) {
  if (shapes.size() < 2)
    return false;
  const auto *prev_sh = shapes[0];
  for (int i = 1; i < shapes.size(); i++) {
    if (IsScalarLike(*prev_sh)) {
      prev_sh = shapes[i];
    } else if (!IsScalarLike(*shapes[i]) && *prev_sh != *shapes[i]) {
      return true;
    }
  }
  return false;
}

bool NeedBroadcasting(span<const TensorShape<>*> shapes) {
  return NeedBroadcastImpl(shapes);
}

bool NeedBroadcasting(span<const TensorListShape<>*> shapes) {
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
  sh = shape_cat(TensorShape<>(std::vector<int64_t>(ndim - sh.size(), 1)), sh);
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
