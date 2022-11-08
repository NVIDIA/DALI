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
namespace expr {

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
  if (extent != 1 && extent != target_value) {
    if (target_value != 1)
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

bool NeedBroadcasting(span<const TensorShape<>*> shapes) {
  int ndim = BroadcastNdim(shapes);
  for (int rev_d = 0; rev_d < ndim; rev_d++) {
    int64_t target = -1;
    for (int i = 0; i < shapes.size(); i++) {
      auto &sh = *shapes[i];
      int64_t extent = rev_d < sh.sample_dim() ? sh[sh.sample_dim() - 1 - rev_d] : 1;
      if (target == -1)
        target = extent;
      else if (target != extent)
        return true;  // not equivalent shapes
    }
  }
  return false;  // all shapes equivalent (equal except leading 1s)
}

bool NeedBroadcasting(span<const TensorListShape<>*> shapes) {
  if (shapes.size() < 2)
    return false;
  int ndim = BroadcastNdim(shapes);
  int nsamples = shapes[0]->num_samples();
  for (int i = 1; i < shapes.size(); i++)
    assert(nsamples == shapes[i]->num_samples());

  for (int s = 0; s < nsamples; s++) {
    for (int rev_d = 0; rev_d < ndim; rev_d++) {
      int target = -1;
      for (int i = 0; i < shapes.size(); i++) {
        auto sh = shapes[i]->tensor_shape_span(s);
        int ndim = shapes[i]->sample_dim();
        int64_t extent = rev_d < ndim ? sh[ndim - 1 - rev_d] : 1;
        if (target == -1)
          target = extent;
        else if (target != extent)
          return true;  // not equivalent shapes
      }
    }
  }
  return false;  // all shapes equivalent (equal except leading 1s)
}

TensorShape<> StridesForBroadcasting(const TensorShape<> &out_sh, const TensorShape<> &in_sh,
                                     const TensorShape<> &in_strides) {
  TensorShape<> strides;
  assert(in_sh.sample_dim() == in_strides.sample_dim());
  assert(in_sh.sample_dim() <= out_sh.sample_dim());
  int out_ndim = out_sh.sample_dim();
  int in_ndim = in_sh.sample_dim();
  assert(in_ndim == in_strides.sample_dim());
  assert(in_ndim <= out_ndim);
  strides.shape.resize(out_ndim, 0);

  for (int i = 0; i < in_ndim; i++) {
    int in_i = in_ndim - i - 1;
    int out_i = out_ndim - i - 1;
    assert(in_sh[in_i] == out_sh[out_i] || in_sh[in_i] == 1);
    if (in_sh[in_i] == out_sh[out_i]) {
      strides[out_i] = in_strides[in_i];
    } else {
      assert(in_sh[in_i] == 1);
      strides[out_i] = 0;
    }
  }
  return strides;
}

void ExpandToNDims(TensorShape<> &sh, int ndim) {
  assert(sh.sample_dim() <= ndim);
  if (sh.sample_dim() == ndim)
    return;
  sh = shape_cat(TensorShape<>(std::vector<int64_t>(ndim - sh.sample_dim(), 1)), sh);
}

void SimplifyShapesForBroadcasting(span<TensorShape<> *> shapes) {
  int n = shapes.size();
  SmallVector<TensorShape<>, 3> outs;
  outs.resize(n);

  int ndim = shapes[0]->sample_dim();
  for (int i = 1; i < n; i++)
    if (static_cast<int>(shapes[i]->sample_dim()) > ndim)
      ndim = shapes[i]->sample_dim();

  auto get = [&](int shape, int dim) -> int64_t {
    auto &s = *shapes[shape];
    dim -= ndim - s.sample_dim();  // add leading unit dims
    if (dim < 0)
      return 1;  // implicit unit dim
    assert(dim >= 0 && dim < s.sample_dim());
    return s[dim];
  };

  // We skip dimensions where all operands have extent 1
  auto should_skip = [&](int d) {
    for (int i = 0; i < n; i++)
      if (get(i, d) != 1)
        return false;
    return true;
  };

  SmallVector<int64_t, 3> volumes;
  volumes.resize(n, 1);

  int group_start = 0;

  // Can collapse current dimension if shares the same condition as the current group.
  // The condition is either extent 1 or not.
  auto can_collapse = [&](int d) {
    for (int i = 0; i < n; i++) {
      bool prev_b = get(i, group_start) == 1;
      bool curr_b = get(i, d) == 1;
      if (prev_b != curr_b)
        return false;
    }
    return true;
  };

  // Closes the group, collapsing all the dims into one
  auto add_group = [&](int d) {
    group_start = d;
    for (int i = 0; i < n; i++) {
      outs[i].shape.push_back(volumes[i]);
      if (d < ndim)
        volumes[i] = get(i, d);
    }
  };

  int d = 0;
  for (; d < ndim; d++) {
    if (!should_skip(d))
      break;
  }

  if (d < ndim) {
    group_start = d;
    for (int i = 0; i < n; i++)
      volumes[i] *= get(i, d);

    for (d++; d < ndim; d++) {
      if (should_skip(d))
        continue;
      if (can_collapse(d)) {
        for (int i = 0; i < n; i++)
          volumes[i] *= get(i, d);
        continue;
      }
      add_group(d);
    }
    add_group(d);
  }

  for (int i = 0; i < n; i++) {
    if (volume(outs[i]) == 1) {
      outs[i] = {};
    }
  }

  for (int i = 0; i < n; i++) {
    *shapes[i] = outs[i];
  }
}

void SimplifyShapesForBroadcasting(TensorShape<> &a, TensorShape<> &b) {
  std::array<TensorShape<>*, 2> arr = {&a, &b};
  SimplifyShapesForBroadcasting(make_span(arr));
}

void SimplifyShapesForBroadcasting(TensorShape<> &a, TensorShape<> &b, TensorShape<>& c) {
  std::array<TensorShape<>*, 3> arr = {&a, &b, &c};
  SimplifyShapesForBroadcasting(make_span(arr));
}

void CheckBroadcastingSimplifiedDim(int ndim) {
  if (ndim > ARITHM_OPS_MAX_DIM) {
    DALI_FAIL(make_string(
          "Broadcasting pattern too complex. Can't operate with simplified shapes with "
          "more than 6 groups of dimensions. Got ", ndim, " groups. For more details "
          "see https://docs.nvidia.com/deeplearning/dali/user-guide/docs/math.html"));
  }
}

}  // namespace expr
}  // namespace dali
