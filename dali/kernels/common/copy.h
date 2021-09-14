// Copyright (c) 2019-2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef DALI_KERNELS_COMMON_COPY_H_
#define DALI_KERNELS_COMMON_COPY_H_

#include <cuda_runtime.h>
#include <cstring>
#include <utility>
#include "dali/core/traits.h"
#include "dali/core/mm/memory.h"
#include "dali/core/backend_tags.h"
#include "dali/core/tensor_view.h"

namespace dali {
namespace kernels {

template <typename StorageOut, typename StorageIn>
void copy(void* out, const void* in, std::size_t N, cudaStream_t stream = 0) {
  if (!is_gpu_accessible<StorageOut>::value) {
    if (is_cpu_accessible<StorageIn>::value) {
      if (is_gpu_accessible<StorageIn>::value)
        CUDA_CALL(cudaStreamSynchronize(stream));  // or cudaDeviceSynchronize?
      std::memcpy(out, in, N);
    } else {
      CUDA_CALL(cudaMemcpyAsync(out, in, N, cudaMemcpyDeviceToHost, stream));
    }
  } else {
    if (is_gpu_accessible<StorageIn>::value) {
      CUDA_CALL(cudaMemcpyAsync(out, in, N, cudaMemcpyDeviceToDevice, stream));
    } else {
      CUDA_CALL(cudaMemcpyAsync(out, in, N, cudaMemcpyHostToDevice, stream));
    }
  }
}

template <typename StorageOut, typename TOut, int NDimOut,
          typename StorageIn, typename TIn, int NDimIn>
void copy(const TensorView<StorageOut, TOut, NDimIn>& out,
          const TensorView<StorageIn, TIn, NDimOut>& in, cudaStream_t stream = 0) {
  static_assert(sizeof(TOut) == sizeof(TIn), "Tensor elements must be of equal size!");
  static_assert(!std::is_const<TOut>::value, "Cannot copy to a tensor of const elements!");
  assert(in.shape == out.shape);
  copy<StorageOut, StorageIn>(out.data, in.data, in.num_elements() * sizeof(TOut), stream);
}

/**
 * @brief Copy in to out, merging contiguous samples.
 *
 * Contiguous samples are merged to reduce number of copies issued and, at the same time,
 * allow for non-contiguous data.
 * The function does not do full coalescing - samples are not sorted to discover order
 *
 * This function has very relaxed shape verification - samples that are stored in contiguous
 * memory are merged and those groups need to have the same total length. Empty samples are skipped
 * and are not treated as discontinuities.
 */
template <typename StorageOut, typename TOut, int NDimOut,
          typename StorageIn, typename TIn, int NDimIn>
void copy(const TensorListView<StorageOut, TOut, NDimOut> &out,
          const TensorListView<StorageIn, TIn, NDimIn> &in,
          cudaStream_t stream = 0) {
  static_assert(sizeof(TOut) == sizeof(TIn), "Tensor elements must be of equal size!");
  static_assert(!std::is_const<TOut>::value, "Cannot copy to a tensor of const elements!");
  int M = in.num_samples();
  int N = out.num_samples();

  TOut *out_start = nullptr;
  const TIn *in_start = nullptr;
  int64_t out_len = 0, in_len = 0;
  int i, o;
  for (i = 0, o = 0; i < M && o < N;) {
    int64_t in_sample_size  = volume(in.shape.tensor_shape_span(i));
    int64_t out_sample_size = volume(out.shape.tensor_shape_span(o));
    if (in_sample_size == 0) {
      i++;
      continue;
    }
    if (out_sample_size == 0) {
      o++;
      continue;
    }
    if (in.data[i] != in_start + in_len || out.data[o] != out_start + out_len) {
      // discontinuity detected
      if (in_len < out_len) {
        assert(in.data[i] != in_start + in_len);
        i++;
        continue;
      }
      if (out_len < in_len) {
        assert(out.data[o] != out_start + out_len);
        o++;
        continue;
      }
      assert(in_len == out_len && "Groups of contiguous samples must have equal length");
      if (out_len > 0)
        copy<StorageOut, StorageIn>(out_start, in_start, out_len * sizeof(TOut), stream);
      out_start = out.data[o];
      in_start = in.data[i];
      in_len = 0;
      out_len = 0;
    }
    in_len  += in_sample_size;
    out_len += out_sample_size;
    i++;
    o++;
  }

  for (; i < M; i++) {
    assert(in.data[i] == in_start + in_len);
    int64_t in_sample_size  = volume(in.shape.tensor_shape_span(i));
    in_len += in_sample_size;
  }

  for (; o < N; o++) {
    assert(in.data[o] == out_start + out_len);
    int64_t out_sample_size  = volume(out.shape.tensor_shape_span(o));
    out_len += out_sample_size;
  }

  assert(i == M && o == N);
  assert(in_len == out_len && "Groups of contiguous samples must have equal length");

  if (out_len > 0)
    copy<StorageOut, StorageIn>(out_start, in_start, out_len * sizeof(TOut), stream);
}

/**
 * Copies input TensorView and returns the output.
 * @tparam DstKind Requested memory kind of the output TensorView.
 *                 According to this parameter, StorageBackend of output TensorView will be determined
 * @tparam NonconstT utility parameter, do not specify (leave default)
 * @return The output consists of new TensorView along with pointer to its memory (as the TensorView doesn't own any)
 */
template<typename DstKind, typename SrcBackend, typename T, int ndims>
std::pair<TensorView<kind2storage_t<DstKind>, dali::remove_const_t<T>, ndims>,
          mm::uptr<dali::remove_const_t<T>>>
copy(const TensorView <SrcBackend, T, ndims> &src) {
  auto mem = mm::alloc_raw_unique<dali::remove_const_t<T>, DstKind>(volume(src.shape));
  auto tv = make_tensor<kind2storage_t<DstKind>, ndims>(mem.get(), src.shape);
  kernels::copy(tv, src);
  return std::make_pair(tv, std::move(mem));
}

}  // namespace kernels
}  // namespace dali

#endif  // DALI_KERNELS_COMMON_COPY_H_
