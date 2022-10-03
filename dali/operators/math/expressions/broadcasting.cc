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

bool HasAnyZeroVolume(span<const TensorShape<>*> shapes) {
  for (int i = 0; i < shapes.size(); i++) {
    if (volume(*shapes[i]) == 0) {
      return true;
    }
  }
  return false;
}

bool HasAnyZeroVolume(span<const TensorListShape<>*> shapes) {
  for (int i = 0; i < shapes.size(); i++) {
    for (int sample_idx = 0; sample_idx < shapes[i]->num_samples(); sample_idx++) {
      if (shapes[i]->tensor_size(sample_idx) == 0) {
        return true;
      }
    }
  }
  return false;
}

void CheckNonZeroVolume(span<const TensorShape<>*> shapes) {
  for (int i = 0; i < shapes.size(); i++) {
    if (volume(*shapes[i]) == 0) {
      DALI_FAIL(make_string("Can't broadcast shapes with zero volume. "
                            "Got a shape with zero volume: ",
                            shapes[i]));
    }
  }
}

void CheckNonZeroVolume(span<const TensorListShape<>*> shapes) {
  for (int i = 0; i < shapes.size(); i++) {
    for (int sample_idx = 0; sample_idx < shapes[i]->num_samples(); sample_idx++) {
      if (shapes[i]->tensor_size(sample_idx) == 0) {
        DALI_FAIL(make_string("Can't broadcast shapes with zero volume. "
                              "Got a shape with zero volume in sample_idx=",
                              sample_idx,  ": ", shapes[i]));
      }
    }
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
 * @param target_value extent value to update (in an output shape)
 * @param shape one of input shapes
 * @param rev_d one of the dimension indices, counting from right to the left
 * @return true if val could be updated (broadcastable)
 * @return false if val could not be updated (non broadcastable)
 */
template <typename ShapeLike>
bool TryBroadcastShapeImpl(int64_t& target_value, const ShapeLike& shape, int rev_d) {
  int ndim = shape.size();
  auto extent = rev_d < ndim ? shape[ndim - 1 - rev_d] : 1;
  assert(extent > 0);
  assert(target_value > 0);
  if (extent > 1 && extent != target_value) {
    if (target_value > 1)
      return false;
    target_value = extent;
  }
  return true;
}

std::string PrintShape(const TensorShape<>& shape, int sample_idx, int rev_d) {
  (void) sample_idx;
  return make_string(to_string(shape), "(d=", shape.sample_dim() - rev_d - 1, ")");
}

std::string PrintShape(const TensorListShape<>& shape, int sample_idx, int rev_d) {
  return make_string(to_string(shape[sample_idx]), " (d=", shape.sample_dim() - rev_d - 1,
                     ", belonging to sample_idx=", sample_idx, ")");
}

template <typename Shapes>
std::string BroadcastErrorMessage(const Shapes& shapes,
                                  int sample_idx, int rev_d) {
  std::stringstream ss;
  ss << "Can't broadcast shapes: ";
  for (int j = 0; j < shapes.size(); j++) {
    ss << "\n" << PrintShape(*shapes[j], sample_idx, rev_d);
    if (j == shapes.size() - 1)
      ss << "\n";
  }
  return ss.str();
}

void BroadcastShape(TensorShape<>& result, span<const TensorShape<>*> shapes) {
  CheckNonZeroVolume(shapes);
  DALI_ENFORCE(shapes.size() >= 1);
  if (shapes.size() == 1) {
    result = *shapes[0];
    return;
  }

  int ndim = BroadcastNdim(shapes);
  result.resize(ndim);

  // We align shapes to the right for comparison,
  // this is why we count d from the right to the left
  for (int rev_d = 0; rev_d < ndim; rev_d++) {
    auto &target_val = result[ndim - 1 - rev_d];
    target_val = 1;
    for (int i = 0; i < shapes.size(); i++) {
      if (!TryBroadcastShapeImpl(target_val, *shapes[i], rev_d)) {
        DALI_FAIL(BroadcastErrorMessage(shapes, 0, rev_d));
      }
    }
  }
}

void BroadcastShape(TensorListShape<>& result, span<const TensorListShape<>*> shapes) {
  CheckNonZeroVolume(shapes);
  DALI_ENFORCE(shapes.size() >= 1);
  if (shapes.size() == 1) {
    result = *shapes[0];
    return;
  }
  CheckNumSamples(shapes);
  int ndim = BroadcastNdim(shapes);
  int nsamples = shapes[0]->num_samples();
  result.resize(nsamples, ndim);

  for (int sample_idx = 0; sample_idx < nsamples; sample_idx++) {
    auto target_sample = result.tensor_shape_span(sample_idx);
    for (int rev_d = 0; rev_d < ndim; rev_d++) {
      auto &target_val = target_sample[target_sample.size() - 1 - rev_d];
      target_val = 1;
      for (int i = 0; i < shapes.size(); i++) {
        if (!TryBroadcastShapeImpl(target_val, shapes[i]->tensor_shape_span(sample_idx), rev_d)) {
          DALI_FAIL(BroadcastErrorMessage(shapes, sample_idx, rev_d));
        }
      }
    }
  }
}

bool CanBroadcast(span<const TensorShape<>*> shapes) {
  if (HasAnyZeroVolume(shapes)) {
    return false;
  }
  if (shapes.size() < 2) {
    return true;
  }
  int ndim = BroadcastNdim(shapes);
  for (int rev_d = 0; rev_d < ndim; rev_d++) {
    int64_t target_value = 1;
    for (int i = 0; i < shapes.size(); i++) {
      if (!TryBroadcastShapeImpl(target_value, *shapes[i], rev_d))
        return false;
    }
  }
  return true;
}

bool CanBroadcast(span<const TensorListShape<>*> shapes) {
  if (HasAnyZeroVolume(shapes)) {
    return false;
  }
  if (shapes.size() < 2) {
    return true;
  }
  int ndim = BroadcastNdim(shapes);
  int nsamples = shapes[0]->num_samples();
  for (int sample_idx = 0; sample_idx < nsamples; sample_idx++) {
    for (int rev_d = 0; rev_d < ndim; rev_d++) {
      int64_t target_value = 1;
      for (int i = 0; i < shapes.size(); i++) {
        if (!TryBroadcastShapeImpl(target_value, shapes[i]->tensor_shape_span(sample_idx), rev_d))
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
  assert(in_sh.sample_dim() == in_strides.sample_dim());
  assert(in_sh.sample_dim() <= out_sh.sample_dim());
  int out_ndim = out_sh.sample_dim();
  strides.shape.resize(out_ndim, 0);
  for (int i = (out_ndim - in_sh.sample_dim()); i < out_ndim; i++) {
    assert(in_sh[i] == out_sh[i] || in_sh[i] == 1);
    if (in_sh[i] == out_sh[i])
      strides[i] = in_strides[i];
    else
      strides[i] = 0;
  }
  return strides;
}

void ExpandToNDims(TensorShape<> &sh, int ndim) {
  assert(sh.sample_dim() <= ndim);
  if (sh.sample_dim() == ndim)
    return;
  sh = shape_cat(TensorShape<>(std::vector<int64_t>(ndim - sh.sample_dim(), 1)), sh);
}

void SimplifyShapesForBroadcasting(span<TensorShape<>*> shapes) {
  if (shapes.size() < 2)
    return;
  // First, if needed expand dimensions
  int full_ndim = shapes[0]->sample_dim();
  for (int i = 1; i < shapes.size(); i++) {
    full_ndim = std::max(full_ndim, shapes[i]->sample_dim());
  }
  for (auto *sh : shapes) {
    ExpandToNDims(*sh, full_ndim);
  }

  int i = 0;
  SmallVector<std::pair<int, int>, 5> group_dims;

  auto all_same = [&shapes](int dim) {
    auto extent = (*shapes[0])[dim];
    for (int k = 1; k < shapes.size(); k++) {
      if (extent != (*shapes[k])[dim]) {
        return false;
      }
    }
    return true;
  };

  while (i < full_ndim) {
    if (!all_same(i)) {
      i++;
      continue;
    }

    int j = i;
    for (; j < full_ndim; j++) {
      if (!all_same(j)) {
        break;
      }
    }
    if (i < j) {
      group_dims.emplace_back(i, j - i);
    }
    i = j;
  }

  for (auto *sh : shapes) {
    *sh = collapse_dims(*sh, group_dims);
  }
}

void SimplifyShapesForBroadcasting(TensorShape<> &a, TensorShape<> &b) {
  std::array<TensorShape<>*, 2> arr = {&a, &b};
  return SimplifyShapesForBroadcasting(make_span(arr));
}

void SimplifyShapesForBroadcasting(TensorShape<> &a, TensorShape<> &b, TensorShape<>& c) {
  std::array<TensorShape<>*, 3> arr = {&a, &b, &c};
  return SimplifyShapesForBroadcasting(make_span(arr));
}


}  // namespace dali
