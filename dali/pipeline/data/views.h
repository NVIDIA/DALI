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

#ifndef  DALI_PIPELINE_DATA_VIEWS_H_
#define  DALI_PIPELINE_DATA_VIEWS_H_

#include <string>
#include <vector>
#include "dali/kernels/tensor_view.h"
#include "dali/kernels/backend_tags.h"
#include "dali/pipeline/data/tensor_list.h"
#include "dali/pipeline/data/tensor.h"

namespace dali {
namespace detail {

/// @brief Maps DALI Backend to dali::kernels storage backend.
template <typename Backend>
struct storage_tag_map;

template <>
struct storage_tag_map<CPUBackend> {
  using type = kernels::StorageCPU;
};

template <>
struct storage_tag_map<GPUBackend> {
  using type = kernels::StorageGPU;
};

template <typename Backend>
using storage_tag_map_t = typename storage_tag_map<Backend>::type;

template <int ndim, typename ShapeType>
void enforce_dim_in_view(const ShapeType &shape) {
  if (ndim != kernels::DynamicDimensions) {
    DALI_ENFORCE(shape.sample_dim() == ndim,
             "Input with dimension (" + to_string(shape.sample_dim())
             + ") cannot be converted to dimension (" + to_string(ndim) + ").");
  }
}

}  // namespace detail

/// @brief Returns an equivalent tensor shape for a dense, uniform tensor list.
/// @return Tensor shape with outermost dimension corresponding to samples.
/// @remarks If the argument is not a dense tensor, an error is raised.
template <int ndim, typename Backend>
kernels::TensorShape<ndim> get_tensor_shape(const TensorList<Backend> &tl) {
  DALI_ENFORCE(tl.IsDenseTensor(), "Uniform, dense tensor expected");
  if (ndim != kernels::DynamicDimensions) {
    DALI_ENFORCE(tl.shape().sample_dim() + 1 == ndim,
    "Input has a wrong number of dimensions!\n"
    "Hint: Converting tensor list to a tensor adds extra dimension");
  }
  int dim = (ndim != kernels::DynamicDimensions) ? ndim : tl.shape().sample_dim() + 1;
  auto out = shape_cat(tl.ntensor(), tl.tensor_shape(0));
  return kernels::convert_dim<ndim>(out);
}

template <typename T, int ndim = kernels::DynamicDimensions, typename Backend>
kernels::TensorListView<detail::storage_tag_map_t<Backend>, T, ndim>
view(TensorList<Backend> &data) {
  if (data.ntensor() == 0)
    return {};
  using U = std::remove_const_t<T>;
  detail::enforce_dim_in_view<ndim>(data.shape());
  return { data.template mutable_data<U>(), kernels::convert_dim<ndim>(data.shape()) };
}


template <typename T, int ndim = kernels::DynamicDimensions, typename Backend>
kernels::TensorListView<detail::storage_tag_map_t<Backend>, T, ndim>
view(const TensorList<Backend> &data) {
  static_assert(std::is_const<T>::value,
                "Cannot create a non-const view of a `const TensorList<>`. "
                "Missing `const` in T?");
  if (data.ntensor() == 0)
    return {};
  using U = std::remove_const_t<T>;
  detail::enforce_dim_in_view<ndim>(data.shape());
  return { data.template data<U>(), kernels::convert_dim<ndim>(data.shape()) };
}

template <typename T, int ndim = kernels::DynamicDimensions, typename Backend>
kernels::TensorView<detail::storage_tag_map_t<Backend>, T, ndim>
view(Tensor<Backend> &data) {
  if (data.shape().empty())
    return {};
  using U = std::remove_const_t<T>;
  detail::enforce_dim_in_view<ndim>(data.shape());
  return { data.template mutable_data<U>(),  kernels::convert_dim<ndim>(data.shape()) };
}

template <typename T, int ndim = kernels::DynamicDimensions, typename Backend>
kernels::TensorView<detail::storage_tag_map_t<Backend>, T, ndim>
view_as_tensor(Tensor<Backend> &data) {
  return view<T, ndim>(data);
}

template <typename T, int ndim = kernels::DynamicDimensions, typename Backend>
kernels::TensorView<detail::storage_tag_map_t<Backend>, T, ndim>
view_as_tensor(TensorList<Backend> &data) {
  if (data.ntensor() == 0)
    return {};
  using U = std::remove_const_t<T>;
  return { data.template mutable_data<U>(), get_tensor_shape<ndim>(data) };
}

template <typename T, int ndim = kernels::DynamicDimensions, typename Backend>
kernels::TensorView<detail::storage_tag_map_t<Backend>, T, ndim>
view(const Tensor<Backend> &data) {
  static_assert(std::is_const<T>::value,
                "Cannot create a non-const view of a `const Tensor<>`. "
                "Missing `const` in T?");
  if (data.shape().empty())
    return {};
  using U = std::remove_const_t<T>;
  detail::enforce_dim_in_view<ndim>(data.shape());
  return { data.template data<U>(), kernels::convert_dim<ndim>(data.shape()) };
}

template <typename T, int ndim = kernels::DynamicDimensions, typename Backend>
kernels::TensorView<detail::storage_tag_map_t<Backend>, T, ndim>
view_as_tensor(const Tensor<Backend> &data) {
  return view<T, ndim>(data);
}

template <typename T, int ndim = kernels::DynamicDimensions, typename Backend>
kernels::TensorView<detail::storage_tag_map_t<Backend>, T, ndim>
view_as_tensor(const TensorList<Backend> &data) {
  static_assert(std::is_const<T>::value,
                "Cannot create a non-const view of a `const TensorList<>`. "
                "Missing `const` in T?");
  if (data.ntensor() == 0)
    return {};
  using U = std::remove_const_t<T>;
  return { data.template data<U>(), get_tensor_shape<ndim>(data) };
}

}  // namespace dali

#endif  // DALI_PIPELINE_DATA_VIEWS_H_
