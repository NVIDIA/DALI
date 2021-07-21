// Copyright (c) 2020-2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef DALI_KERNELS_REDUCE_REDUCE_TEST_H_
#define DALI_KERNELS_REDUCE_REDUCE_TEST_H_

#include <memory>
#include <utility>
#include "dali/core/span.h"
#include "dali/core/tensor_view.h"
#include "dali/kernels/reduce/online_reducer.h"
#include "dali/kernels/common/utils.h"

namespace dali {
namespace kernels {

template <typename Out, typename Reduction, typename T>
Out RefReduce(const T *in, int64_t n, int64_t stride, const Reduction &R) {
  switch (n) {
    case 0:
      return R.template neutral<Out>();
    case 1:
      return in[0];
    default: {
      if (n <= 128) {
        Out acc = R.template neutral<Out>();
        for (int64_t idx = 0; idx < n; idx++)
          R(acc, in[idx * stride]);
        return acc;
      }
      int64_t a = n / 2;
      int64_t b = n - a;
      Out out = RefReduce<Out>(in, a, stride, R);
      R(out, RefReduce<Out>(in + a * stride, b, stride, R));
      return out;
    }
  }
}


template <typename Out, typename Reduction, typename T>
Out RefReduce(span<T> in, const Reduction &R) {
  return RefReduce<Out>(in.data(), in.size(), 1, R);
}


template <typename OnlineReducer, typename In>
void RefReduceStrided(OnlineReducer &R, const In *in, const int64_t *in_stride,
               const int64_t *in_extent, int dim) {
  if (dim == 0) {
    R.add(*in);
  } else if (dim == 1) {
    const int64_t n = in_extent[0];
    const int64_t stride = in_stride[0];
    for (int64_t i = 0; i < n; i++) {
      R.add(*in);
      in += stride;
    }
  } else {
    for (int64_t i = 0; i < in_extent[0]; i++) {
      RefReduceStrided(R, in, in_stride+1, in_extent+1, dim - 1);
      in += in_stride[0];
    }
  }
}


template <typename Reduction, typename Out, typename In>
void RefReduceAxes(Out *out, const In *in,
                   const int64_t *reduced_stride, const int64_t *reduced_extent,
                   int reduced_dim,
                   const int64_t *non_reduced_stride, const int64_t *non_reduced_extent,
                   int non_reduced_dim, Reduction R = {}) {
  if (non_reduced_dim == 0) {
    // only reduced dimensions left - let's reduce them and store the value
    OnlineReducer<Out, Reduction> red;
    red = {};
    RefReduceStrided(red, in, reduced_stride, reduced_extent, reduced_dim);
    *out = red.result();
  } else {
    // traverse remaining non-reduced dimensions

    // output stride is a plain product of remaining inner extents
    int64_t out_stride = non_reduced_dim > 1
      ? volume(make_span(non_reduced_extent + 1, non_reduced_dim - 1))
      : 1;
    for (int64_t i = 0; i < non_reduced_extent[0]; i++) {
      RefReduceAxes(out, in, reduced_stride, reduced_extent, reduced_dim,
                    non_reduced_stride + 1, non_reduced_extent + 1, non_reduced_dim - 1, R);
      in += non_reduced_stride[0];
      out += out_stride;
    }
  }
}


template <typename Out, typename In, typename Reduction>
void RefReduce(const TensorView<StorageCPU, Out> &out, const TensorView<StorageCPU, In> &in,
               span<const int> axes, bool keep_dims, Reduction R = {}) {
  uint64_t mask = 0;
  for (int idx : axes) {
    mask |= 1uL << idx;
  }

  SmallVector<int, 6> reduced_dims, non_reduced_dims;
  auto strides = GetStrides(in.shape.shape);
  SmallVector<int64_t, 6> reduced_strides, non_reduced_strides;
  SmallVector<int64_t, 6> reduced_extents, non_reduced_extents;

  // factorize the tensor into reduced and non-reduced parts

  int ndim = strides.size();
  for (int i = 0, o = 0; i < ndim; i++) {
    if (mask & (1_u64 << i)) {
      if (keep_dims) {
        assert(out.shape[o] == 1);
        o++;
      }
      reduced_strides.push_back(strides[i]);
      reduced_extents.push_back(in.shape[i]);
    } else {
      assert(out.shape[o] == in.shape[i]);
      o++;
      non_reduced_strides.push_back(strides[i]);
      non_reduced_extents.push_back(in.shape[i]);
    }
  }

  RefReduceAxes(out.data, in.data,
    reduced_strides.data(), reduced_extents.data(), reduced_extents.size(),
    non_reduced_strides.data(), non_reduced_extents.data(), non_reduced_extents.size(), R);
}

template <typename Out, typename In, typename Reduction>
void RefReduce(const TensorListView<StorageCPU, Out> &out,
               const TensorListView<StorageCPU, In> &in,
               span<const int> axes, bool keep_dims, bool batch, Reduction R = {}) {
  if (batch && in.num_samples() > 1) {
    assert(out.shape.num_samples() == 1);
    auto s_shape = uniform_list_shape(in.num_samples(), out.shape[0]);
    unique_ptr<Out[]> s_data(new Out[s_shape.num_elements()]);
    auto reduced_samples = make_tensor_list_cpu(s_data.get(), std::move(s_shape));
    RefReduce(reduced_samples, in, axes, keep_dims, false, R);
    int64_t n = out.shape[0].num_elements();
    int N = in.num_samples();
    for (int64_t j = 0; j < n; j++) {
      OnlineReducer<Out, Reduction> red;
      red.reset();
      for (int i = 0; i < N; i++) {
        red.add(reduced_samples.data[i][j], R);
      }
      out.data[0][j] = red.result();
    }
  } else {
    assert(out.num_samples() == in.num_samples());
    for (int i = 0; i < in.num_samples(); i++) {
      RefReduce(out[i], in[i], axes, keep_dims, R);
    }
  }
}

}  // namespace kernels
}  // namespace dali

#endif  // DALI_KERNELS_REDUCE_REDUCE_TEST_H_
