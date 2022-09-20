// Copyright (c) 2019-2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include <type_traits>
#include <utility>
#include <vector>
#include "dali/core/backend_tags.h"
#include "dali/core/tensor_shape.h"
#include "dali/core/tensor_view.h"
#include "dali/pipeline/data/sample_view.h"
#include "dali/pipeline/data/tensor.h"
#include "dali/pipeline/data/tensor_list.h"

namespace dali {
namespace detail {

/**
 * @brief Maps DALI Backend to dali::kernels storage backend.
 */
template <typename Backend>
struct storage_tag_map;

template <>
struct storage_tag_map<CPUBackend> {
  using type = StorageCPU;
};

template <>
struct storage_tag_map<GPUBackend> {
  using type = StorageGPU;
};

template <typename Backend>
using storage_tag_map_t = typename storage_tag_map<Backend>::type;

template <int ndim, typename ShapeType>
void enforce_dim_in_view(const ShapeType &shape) {
  if (ndim != DynamicDimensions) {
    DALI_ENFORCE(shape.sample_dim() == ndim,
             "Input with dimension (" + to_string(shape.sample_dim())
             + ") cannot be converted to dimension (" + to_string(ndim) + ").");
  }
}

}  // namespace detail


template <typename T, int ndim = DynamicDimensions, typename Backend>
TensorView<detail::storage_tag_map_t<Backend>, T, ndim>
view(Tensor<Backend> &data) {
  using U = std::remove_const_t<T>;
  detail::enforce_dim_in_view<ndim>(data.shape());
  return { data.template mutable_data<U>(), convert_dim<ndim>(data.shape()) };
}


template <typename T, int ndim = DynamicDimensions, typename Backend>
TensorView<detail::storage_tag_map_t<Backend>, T, ndim>
view(const Tensor<Backend> &data) {
  static_assert(std::is_const<T>::value,
                "Cannot create a non-const view of a `const Tensor<>`. "
                "Missing `const` in T?");
  using U = std::remove_const_t<T>;
  detail::enforce_dim_in_view<ndim>(data.shape());
  return { data.template data<U>(), convert_dim<ndim>(data.shape()) };
}


/**
 * @name Convert from SampleView carrying runtime type information to statically typed TensorView.
 */
template <typename T, int ndim = DynamicDimensions, typename Backend>
TensorView<detail::storage_tag_map_t<Backend>, T, ndim> view(SampleView<Backend> data) {
  using U = std::remove_const_t<T>;
  detail::enforce_dim_in_view<ndim>(data.shape());
  return {data.template mutable_data<U>(), data.shape()};
}

template <typename T, int ndim = DynamicDimensions, typename Backend>
TensorView<detail::storage_tag_map_t<Backend>, T, ndim> view(ConstSampleView<Backend> data) {
  static_assert(std::is_const<T>::value,
                "Cannot create a non-const view of a `const Tensor<>`. "
                "Missing `const` in T?");
  using U = std::remove_const_t<T>;
  detail::enforce_dim_in_view<ndim>(data.shape());
  return {data.template data<U>(), data.shape()};
}
// @}


template <typename T, int ndim = DynamicDimensions, typename Backend>
TensorListView<detail::storage_tag_map_t<Backend>, T, ndim> view(TensorList<Backend> &data) {
  if (data.num_samples() == 0)
    return {};
  using U = std::remove_const_t<T>;
  const auto &shape = data.shape();
  detail::enforce_dim_in_view<ndim>(shape);

  std::vector<T *> ptrs(shape.num_samples());
  for (int i = 0; i < shape.num_samples(); i++) {
    ptrs[i] = data.template mutable_tensor<U>(i);
  }
  return { std::move(ptrs), convert_dim<ndim>(shape) };
}


template <typename T, int ndim = DynamicDimensions, typename Backend>
TensorListView<detail::storage_tag_map_t<Backend>, T, ndim> view(const TensorList<Backend> &data) {
  static_assert(std::is_const<T>::value,
                "Cannot create a non-const view of a `const TensorList<>`. "
                "Missing `const` in T?");
  if (data.num_samples() == 0)
    return {};
  using U = std::remove_const_t<T>;
  const auto &shape = data.shape();
  detail::enforce_dim_in_view<ndim>(shape);

  std::vector<T *> ptrs(shape.num_samples());
  for (int i = 0; i < shape.num_samples(); i++) {
    ptrs[i] = data.template tensor<U>(i);
  }
  return { std::move(ptrs), convert_dim<ndim>(shape) };
}


template <typename T, int ndim = DynamicDimensions, typename Backend>
TensorListView<detail::storage_tag_map_t<Backend>, T, ndim> reinterpret_view(
    TensorList<Backend> &data) {
  if (data.num_samples() == 0)
    return {};
  detail::enforce_dim_in_view<ndim>(data.shape());
  TensorListView<detail::storage_tag_map_t<Backend>, T, ndim> ret;
  ret.shape = convert_dim<ndim>(data.shape());
  ret.data.resize(ret.shape.num_samples());
  assert(data.type_info().size() >= sizeof(T));
  assert(data.type_info().size() % sizeof(T) == 0);
  for (int i = 0; i < ret.shape.num_samples(); i++) {
    ret.data[i] = static_cast<T*>(data.raw_mutable_tensor(i));
  }
  // If reinterpreting to a smaller type, adjust the inner extent
  if (data.type_info().size() > sizeof(T)) {
    int k = data.type_info().size() / sizeof(T);
    for (int i = 0; i < ret.shape.num_samples(); i++) {
      auto sh = ret.shape.tensor_shape_span(i);
      sh[sh.size() - 1] *= k;
    }
  }
  return ret;
}


template <typename T, int ndim = DynamicDimensions, typename Backend>
TensorListView<detail::storage_tag_map_t<Backend>, T, ndim> reinterpret_view(
    const TensorList<Backend> &data) {
  static_assert(std::is_const<T>::value,
                "Cannot create a non-const view of a `const TensorList<>`. "
                "Missing `const` in T?");
  if (data.num_samples() == 0)
    return {};
  detail::enforce_dim_in_view<ndim>(data.shape());
  TensorListView<detail::storage_tag_map_t<Backend>, T, ndim> ret;
  ret.shape = convert_dim<ndim>(data.shape());
  ret.data.resize(ret.shape.num_samples());
  assert(data.type_info().size() >= sizeof(T));
  assert(data.type_info().size() % sizeof(T) == 0);
  for (int i = 0; i < ret.shape.num_samples(); i++) {
    ret.data[i] = static_cast<T*>(data.raw_tensor(i));
  }
  // If reinterpreting to a smaller type, adjust the inner extent
  if (data.type_info().size() > sizeof(T)) {
    int k = data.type_info().size() / sizeof(T);
    for (int i = 0; i < ret.shape.num_samples(); i++) {
      auto sh = ret.shape.tensor_shape_span(i);
      sh[sh.size() - 1] *= k;
    }
  }
  return ret;
}


template <typename Backend>
SampleView<Backend>
sample_view(Tensor<Backend> &data) {
  return { data.raw_mutable_data(), data.shape(), data.type() };
}


template <typename Backend>
ConstSampleView<Backend>
const_sample_view(const Tensor<Backend> &data) {
  return { data.raw_data(), data.shape(), data.type() };
}

}  // namespace dali

#endif  // DALI_PIPELINE_DATA_VIEWS_H_
