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

}  // namespace detail

template <int ndim, typename Backend>
kernels::TensorListShape<ndim> list_shape(const TensorList<Backend> &tl) {
  const auto &tshape = tl.tensor_shape(0);
  if (ndim != kernels::DynamicDimensions)
    DALI_ENFORCE((int)tshape.size() == ndim, "Input has a wrong number of dimensions");
  return kernels::convert_dim<ndim>(kernels::TensorListShape<>(tl.shape()));
}

/// @brief Returns an equivalent tensor shape for a dense, uniform tensor list.
/// @return Tensor shape with outermost dimension corresponding to samples.
/// @remarks If the argument is not a dense tensor, an error is raised.
template <int ndim, typename Backend>
kernels::TensorShape<ndim> tensor_shape(const TensorList<Backend> &tl) {
  const auto &tshape = tl.tensor_shape(0);
  DALI_ENFORCE(tl.IsDenseTensor(), "Uniform, dense tensor expected");
  if (ndim != kernels::DynamicDimensions) {
    DALI_ENFORCE((int)tshape.size()+1 == ndim,
    "Input has a wrong number of dimensions!\n"
    "Hint: Converting tensor list to a tensor adds extra dimension");
  }
  int dim = (ndim != kernels::DynamicDimensions) ? ndim : tshape.size()+1;
  kernels::TensorShape<ndim> out;
  out.resize(dim);
  out[0] = tl.ntensor();
  for (int i = 0; i < dim-1; i++) {
    out[i+1] = tshape[i];
  }
  return out;
}


/// @brief Returns an equivalent tensor list shape for a tensor.
///        Outermost dimension is converted into sample index.
template<int ndim, typename Backend>
kernels::TensorShape<ndim> tensor_shape(const Tensor<Backend> &tl) {
  const auto &tshape = tl.shape();
  kernels::TensorShape<ndim> out;
  int dim = tshape.size();
  if (ndim != kernels::DynamicDimensions) {
    DALI_ENFORCE(dim == ndim,
                 "Input has a wrong number of dimensions:"
                 " (" + to_string(dim) + ") vs (" + to_string(ndim) + ")");
  } else {
    out.resize(dim);
  }
  for (int i = 0; i < dim; i++) {
    out[i] = tshape[i];
  }
  return out;
}

template <typename T, int ndim = kernels::DynamicDimensions, typename Backend>
kernels::TensorListView<detail::storage_tag_map_t<Backend>, T, ndim>
view(TensorList<Backend> &data) {
  if (data.ntensor() == 0)
    return {};
  using U = typename std::remove_const<T>::type;
  return { data.template mutable_data<U>(), list_shape<ndim>(data) };
}


template <typename T, int ndim = kernels::DynamicDimensions, typename Backend>
kernels::TensorListView<detail::storage_tag_map_t<Backend>, T, ndim>
view(const TensorList<Backend> &data) {
  static_assert(std::is_const<T>::value,
                "Cannot create a non-const view of a `const TensorList<>`. "
                "Missing `const` in T?");
  if (data.ntensor() == 0)
    return {};
  using U = typename std::remove_const<T>::type;
  return { data.template data<U>(), list_shape<ndim>(data) };
}

template <typename T, int ndim = kernels::DynamicDimensions, typename Backend>
kernels::TensorView<detail::storage_tag_map_t<Backend>, T, ndim>
view(Tensor<Backend> &data) {
  if (data.shape().empty())
    return {};
  using U = typename std::remove_const<T>::type;
  return { data.template mutable_data<U>(), tensor_shape<ndim>(data) };
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
  using U = typename std::remove_const<T>::type;
  return { data.template mutable_data<U>(), tensor_shape<ndim>(data) };
}

template <typename T, int ndim = kernels::DynamicDimensions, typename Backend>
kernels::TensorView<detail::storage_tag_map_t<Backend>, T, ndim>
view(const Tensor<Backend> &data) {
  static_assert(std::is_const<T>::value,
                "Cannot create a non-const view of a `const Tensor<>`. "
                "Missing `const` in T?");
  if (data.shape().empty())
    return {};
  using U = typename std::remove_const<T>::type;
  return { data.template data<U>(), tensor_shape<ndim>(data) };
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
  using U = typename std::remove_const<T>::type;
  return { data.template data<U>(), tensor_shape<ndim>(data) };
}

template <int ndim>
void to_dims_vec(std::vector<Dims> &dims_vec, const kernels::TensorListShape<ndim> &tls) {
  const int dim = tls.sample_dim();
  const int N = tls.num_samples();
  dims_vec.resize(N);
  for (int i = 0; i < N; i++) {
    dims_vec[i].resize(dim);
    for (int j = 0; j < dim; j++)
      dims_vec[i][j] = tls.tensor_shape_span(i)[j];
  }
}

}  // namespace dali

#endif  // DALI_PIPELINE_DATA_VIEWS_H_
