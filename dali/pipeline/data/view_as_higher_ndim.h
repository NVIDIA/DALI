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

#ifndef  DALI_PIPELINE_DATA_VIEW_AS_HIGHER_NDIM_H_
#define  DALI_PIPELINE_DATA_VIEW_AS_HIGHER_NDIM_H_

#include <utility>
#include <vector>
#include "dali/pipeline/data/views.h"

namespace dali {

/**
 * @brief Inserts `insert_ndim` new dimensions with extent one to a given position in a shape
 */
static TensorListShape<> insert_unit_dims(const TensorListShape<> &sh, int insert_ndim,
                                          int position) {
  int ndim = sh.sample_dim();
  int nsamples = sh.num_samples();

  if (sh.sample_dim() > ndim) {
    throw std::logic_error(
        make_string("Can't view ", sh.sample_dim(), "-D data as ", ndim, "-D"));
  }

  TensorListShape<> new_sh;
  new_sh.resize(nsamples, ndim + insert_ndim);
  for (int s = 0; s < nsamples; s++) {
    auto sample_sh = sh.tensor_shape_span(s);
    auto new_sample_sh = new_sh.tensor_shape_span(s);

    int out_d = 0;
    for (int d = 0; d < ndim; d++) {
      if (d == position)
        out_d += insert_ndim;
      new_sample_sh[out_d++] = sample_sh[d];
    }
    for (int out_d = position; out_d < position + insert_ndim; out_d++) {
      new_sample_sh[out_d] = 1;
    }
  }
  return new_sh;
}

/**
 * @name Obtains a TensorListView with a higher number of dimensions, padding with unit dimensions
 * to the left or the right of each shape
 */
template <typename T, int ndim, typename Backend>
TensorListView<detail::storage_tag_map_t<Backend>, T, ndim> view_as_higher_ndim(
    TensorList<Backend> &data, bool pad_left = true) {
  if (data.num_samples() == 0)
    return {};
  using U = std::remove_const_t<T>;
  auto shape = data.shape();
  int orig_ndim = shape.sample_dim();

  if (orig_ndim > ndim) {
    throw std::invalid_argument(make_string("Expected a shape with fewer than ", ndim,
                                            " dimensions. Got ", orig_ndim, " dimensions."));
  } else if (orig_ndim == ndim) {
    return view<T, ndim, Backend>(data);
  }

  int insert_ndim = ndim - orig_ndim;
  shape = insert_unit_dims(shape, insert_ndim, pad_left ? 0 : orig_ndim);
  detail::enforce_dim_in_view<ndim>(shape);

  std::vector<T *> ptrs(shape.num_samples());
  for (int i = 0; i < shape.num_samples(); i++) {
    ptrs[i] = data.template mutable_tensor<U>(i);
  }
  return { std::move(ptrs), convert_dim<ndim>(shape) };
}


template <typename T, int ndim, typename Backend>
TensorListView<detail::storage_tag_map_t<Backend>, T, ndim> view_as_higher_ndim(
    const TensorList<Backend> &data, bool pad_left = true) {
  static_assert(std::is_const<T>::value,
                "Cannot create a non-const view of a `const TensorList<>`. "
                "Missing `const` in T?");
  if (data.num_samples() == 0)
    return {};
  using U = std::remove_const_t<T>;
  auto shape = data.shape();
  int orig_ndim = shape.sample_dim();

  if (orig_ndim > ndim) {
    throw std::invalid_argument(make_string("Expected a shape with fewer than ", ndim,
                                            " dimensions. Got ", orig_ndim, " dimensions."));
  } else if (orig_ndim == ndim) {
    return view<T, ndim, Backend>(data);
  }

  int insert_ndim = ndim -orig_ndim;
  shape = insert_unit_dims(shape, insert_ndim, pad_left ? 0 : orig_ndim);
  detail::enforce_dim_in_view<ndim>(shape);

  std::vector<T *> ptrs(shape.num_samples());
  for (int i = 0; i < shape.num_samples(); i++) {
    ptrs[i] = data.template tensor<U>(i);
  }
  return { std::move(ptrs), convert_dim<ndim>(shape) };
}
// @}

}  // namespace dali

#endif  // DALI_PIPELINE_DATA_VIEW_AS_HIGHER_NDIM_H_
