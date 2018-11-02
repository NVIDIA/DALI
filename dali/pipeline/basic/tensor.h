// Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
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

#ifndef DALI_PIPELINE_BASIC_TENSOR_H_
#define DALI_PIPELINE_BASIC_TENSOR_H_

#include "dali/common.h"
#include "dali/pipeline/data/tensor.h"

namespace dali {
namespace basic {

template <size_t N>
struct TensorShape {
  explicit TensorShape(const Tensor<CPUBackend>& t) {
    for (size_t i = 0; i < N; i++) {
      shape[i] = t.dim(i);
    }
  }

  explicit TensorShape(const TensorShape& other) {
    for (size_t i = 0; i < N; i++) {
      shape[i] = other.shape[i];
    }
  }

  // Why call for TensorWrapper (subclass) calls this instead of proper copy constructor??
  template <typename Iter>
  explicit TensorShape(Iter it) {
    for (size_t i = 0; i < N; i++) {
      shape[i] = *it;
      ++it;
    }
  }

  const std::array<Index, N>& GetShape() const { return shape; }

 private:
  std::array<Index, N> shape;
};

template <typename T, size_t N>
struct TensorWrapper : public TensorShape<N> {
  TensorWrapper() = delete;
  // TODO(klecki): C++ way instead of Google's. Should probably be fixed
  explicit TensorWrapper(Tensor<CPUBackend>& t)  // NOLINT
      : TensorShape<N>(t), ptr(t.mutable_data<T>()) {
    // TODO(klecki): I think this is not the place for this kind of checks
    DALI_ENFORCE(ptr != nullptr, "Tensor wrapper does not accept nullptr");
  }

  TensorWrapper(const TensorWrapper& t) : TensorShape<N>(t.GetShape().cbegin()), ptr(t.ptr) {
    DALI_ENFORCE(ptr != nullptr, "Tensor wrapper does not accept nullptr");
  }

  template <typename Iter>
  TensorWrapper(T* ptr, Iter it) : TensorShape<N>(it), ptr(ptr) {
    DALI_ENFORCE(ptr != nullptr, "Tensor wrapper does not accept nullptr");
  }

  T* const ptr;
  using type = T;
  using TensorShape<N>::GetShape;
};

template <typename T, size_t N>
struct TensorWrapper<const T, N> : public TensorShape<N> {
  TensorWrapper() = delete;
  explicit TensorWrapper(const Tensor<CPUBackend>& t) : TensorShape<N>(t), ptr(t.data<T>()) {
    DALI_ENFORCE(ptr != nullptr, "Tensor wrapper does not accept nullptr");
  }

  TensorWrapper(const TensorWrapper& t) : TensorShape<N>(t.GetShape().cbegin()), ptr(t.ptr) {
    DALI_ENFORCE(ptr != nullptr, "Tensor wrapper does not accept nullptr");
  }

  template <typename Iter>
  TensorWrapper(const T* ptr, Iter it) : TensorShape<N>(it), ptr(ptr) {
    DALI_ENFORCE(ptr != nullptr, "Tensor wrapper does not accept nullptr");
  }

  const T* const ptr;
  using type = const T;
  using TensorShape<N>::GetShape;
};

template <typename T, size_t N>
struct SequenceWrapper : public TensorWrapper<T, N> {
  static_assert(N > 1, "Tensor is required to have at least 2 dimensions to become sequence");
  SequenceWrapper() = delete;
  explicit SequenceWrapper(const TensorWrapper<T, N>& t)
      : TensorWrapper<T, N>(t),
        sequence_length(t.GetShape()[0]),
        element_offset(CalcOffset(t.GetShape())) {}

  TensorWrapper<T, N - 1> Get(size_t idx) {
    return TensorWrapper<T, N - 1>(ptr + idx * element_offset, GetShape().cbegin() + 1);
  }

  const Index sequence_length;
  const Index element_offset;
  using TensorWrapper<T, N>::GetShape;
  using TensorWrapper<T, N>::ptr;
  using typename TensorWrapper<T, N>::type;

 private:
  Index CalcOffset(const std::array<Index, N>& shape) {
    Index offset = shape[1];
    for (size_t i = 2; i < N; i++) {
      offset *= shape[i];
    }
    return offset;
  }
};

// TODO(klecki) template to add dim?
template <typename T>
struct add_dim {};

template <typename T, size_t N>
struct add_dim<TensorWrapper<T, N>> {
  using type = TensorWrapper<T, N + 1>;
};

template <typename T>
using add_dim_t = typename add_dim<T>::type;

}  // namespace basic
}  // namespace dali

#endif  // DALI_PIPELINE_BASIC_TENSOR_H_
