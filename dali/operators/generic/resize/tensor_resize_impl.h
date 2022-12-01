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

#ifndef DALI_OPERATORS_GENERIC_RESIZE_TENSOR_RESIZE_IMPL_H_
#define DALI_OPERATORS_GENERIC_RESIZE_TENSOR_RESIZE_IMPL_H_

#include <cassert>
#include <memory>
#include <vector>
#include <utility>
#include "dali/pipeline/data/views.h"
#include "dali/pipeline/operator/common.h"
#include "dali/pipeline/operator/operator.h"
#include "dali/pipeline/operator/arg_helper.h"

// #define TENSOR_RESIZE_SUPPORTED_NDIM (2, 3)
// #define TENSOR_RESIZE_SUPPORTED_TYPES (uint8_t, int8_t, uint16_t, int16_t, uint32_t, int32_t, \
//                                        uint64_t, int64_t, float)

#define TENSOR_RESIZE_SUPPORTED_NDIM (2, 3)
#define TENSOR_RESIZE_SUPPORTED_TYPES (uint8_t, float)


namespace dali {
namespace tensor_resize {

static TensorListShape<> AppendUnitDims(const TensorListShape<>& sh, int new_ndim) {
  int ndim = sh.sample_dim();
  int nsamples = sh.num_samples();

  if (sh.sample_dim() > ndim) {
    throw std::logic_error(
        make_string("Can't view ", sh.sample_dim(), "-D data as ", ndim, "-D"));
  }

  TensorListShape<> new_sh;
  new_sh.resize(nsamples, new_ndim);
  for (int s = 0; s < nsamples; s++) {
    auto sample_sh = sh.tensor_shape_span(s);
    auto new_sample_sh = new_sh.tensor_shape_span(s);
    for (int d = 0; d < ndim; d++)
      new_sample_sh[d] = sample_sh[d];
    for (int d = ndim; d < new_ndim; d++)
      new_sample_sh[ndim] = 1;
  }
  return new_sh;
}


template <typename T, int ndim, typename Backend>
TensorListView<detail::storage_tag_map_t<Backend>, T, ndim> view_w_extra_ch_dim(
    TensorList<Backend> &data) {
  if (data.num_samples() == 0)
    return {};
  using U = std::remove_const_t<T>;
  auto shape = data.shape();

  shape = AppendUnitDims(shape, ndim);
  detail::enforce_dim_in_view<ndim>(shape);

  std::vector<T *> ptrs(shape.num_samples());
  for (int i = 0; i < shape.num_samples(); i++) {
    ptrs[i] = data.template mutable_tensor<U>(i);
  }
  return { std::move(ptrs), convert_dim<ndim>(shape) };
}


template <typename T, int ndim, typename Backend>
TensorListView<detail::storage_tag_map_t<Backend>, T, ndim> view_w_extra_ch_dim(
    const TensorList<Backend> &data) {
  static_assert(std::is_const<T>::value,
                "Cannot create a non-const view of a `const TensorList<>`. "
                "Missing `const` in T?");
  if (data.num_samples() == 0)
    return {};
  using U = std::remove_const_t<T>;
  auto shape = data.shape();

  shape = AppendUnitDims(shape, ndim);
  detail::enforce_dim_in_view<ndim>(shape);

  std::vector<T *> ptrs(shape.num_samples());
  for (int i = 0; i < shape.num_samples(); i++) {
    ptrs[i] = data.template tensor<U>(i);
  }
  return { std::move(ptrs), convert_dim<ndim>(shape) };
}

}  // namespace tensor_resize
}  // namespace dali


#endif  // DALI_OPERATORS_GENERIC_RESIZE_TENSOR_RESIZE_IMPL_H_
