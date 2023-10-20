// Copyright (c) 2021-2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <string>
#include <vector>
#include "dali/core/convert.h"
#include "dali/core/static_switch.h"
#include "dali/kernels/slice/slice_gpu.cuh"
#include "dali/kernels/slice/slice_kernel_utils.h"
#include "dali/kernels/transpose/transpose_gpu.h"
#include "dali/operators/reader/numpy_reader_gpu_op.h"
#include "dali/pipeline/data/views.h"

namespace dali {

template <typename T, int Dims>
void NumpyReaderGPU::RunImplTyped(Workspace &ws) {
  auto &output = ws.Output<GPUBackend>(0);
  const auto &out_sh = output.shape();
  int ndim = out_sh.sample_dim();
  int nsamples = out_sh.num_samples();
  bool need_slice = !rois_.empty();
  auto out_view = view<T>(output);
  auto curr_batch = GetCurrBatchView<T, Dims>();
  const auto &dtype = TypeTable::GetTypeInfo<T>();
  int batch_size = GetCurrBatchSize();
  source_data_index_.clear();
  source_data_index_.resize(batch_size);

  CUDA_CALL(cudaStreamWaitEvent(ws.stream(), staging_ready_, 0));

  // Permuted dims, to use for the transposition
  std::array<int, Dims> perm;
  for (int d = 0; d < Dims; d++)
    perm[d] = Dims - 1 - d;

  int nsamples_copy = 0, nsamples_slice = 0, nsamples_transpose = 0, nsamples_slice_transpose = 0;
  for (int i = 0; i < nsamples; i++) {
    if (need_slice_[i] && need_transpose_[i])
      nsamples_slice_transpose++;
    if (need_slice_[i])
      nsamples_slice++;
    if (need_transpose_[i])
      nsamples_transpose++;
    if (!need_slice_[i] && !need_transpose_[i])
      nsamples_copy++;
    source_data_index_[i] = GetSample(i).source_sample_idx;
  }

  if (nsamples_copy) {
    bool is_padded = false;
    for (int i = 0; i < nsamples; ++i) {
      if (source_data_index_[i] != i) {
        is_padded = true;
        break;
      }
    }
    if (nsamples_copy == nsamples && !is_padded) {
      std::swap(output, prefetched_batch_tensors_[curr_batch_consumer_]);
      return;
    }
    for (int i = 0; i < nsamples; i++) {
      if (need_slice_[i] || need_transpose_[i])
        continue;
      auto sz = out_sh.tensor_size(i) * sizeof(T);
      int src_idx = source_data_index_[i];
      sg_.AddCopy(out_view.data[i], curr_batch.data[src_idx], sz);
    }
    sg_.Run(ws.stream());
  }

  TensorListView<StorageGPU, T, Dims> tmp_view;
  if (nsamples_slice_transpose) {
    tmp_buf_sh_.resize(nsamples_slice_transpose, Dims);
    for (int i = 0, j = 0; i < nsamples; i++) {
      if (need_slice_[i] && need_transpose_[i]) {
        tmp_buf_sh_.set_tensor_shape(j, rois_[i].shape);
        j++;
      }
    }
    tmp_buf_.Resize(tmp_buf_sh_, dtype.id());
    tmp_view = view<T, Dims>(tmp_buf_);
  }

  if (nsamples_slice) {
    // TLV used to invoke kernels with a subset of the samples
    TensorListView<StorageGPU, const T, Dims> from;
    from.resize(nsamples_slice);
    TensorListView<StorageGPU, T, Dims> to;
    to.resize(nsamples_slice);

    std::vector<kernels::SliceArgs<T, Dims>> slice_args;
    slice_args.resize(nsamples_slice);
    for (int i = 0, j = 0, k = 0; i < nsamples; i++) {
      if (!need_slice_[i])
        continue;
      auto &args = slice_args[j];
      args.anchor = rois_[i].anchor;
      args.shape = rois_[i].shape;
      args.fill_values.clear();
      args.fill_values.push_back(ConvertSat<T>(fill_value_));

      int src_idx = source_data_index_[i];
      from.data[j] = curr_batch.data[src_idx];
      from.shape.set_tensor_shape(j, curr_batch.shape[src_idx]);

      if (need_transpose_[i]) {
        // slice to an intermediate buffer
        to.data[j] = tmp_view.data[k];
        to.shape.set_tensor_shape(j, tmp_view.tensor_shape(k));
        k++;
      } else {
        // directly to the output
        to.data[j] = out_view.data[i];
        to.shape.set_tensor_shape(j, out_view.shape[i]);
      }
      j++;
    }
    kernels::KernelContext ctx;
    ctx.gpu.stream = ws.stream();
    using Kernel = kernels::SliceGPU<T, T, Dims>;
    kmgr_slice_.Resize<Kernel>(1);
    kmgr_slice_.Setup<Kernel>(0, ctx, from, slice_args);
    kmgr_slice_.Run<Kernel>(0, ctx, to, from, slice_args);
  }

  if (nsamples_transpose) {
    // TLV used to invoke kernels with a subset of the samples
    TensorListView<StorageGPU, const T> from;
    from.resize(nsamples_transpose, ndim);
    TensorListView<StorageGPU, T> to;
    to.resize(nsamples_transpose, ndim);

    for (int i = 0, j = 0, k = 0; i < nsamples; i++) {
      if (!need_transpose_[i])
        continue;
      if (need_slice_[i]) {
        // from sliced data
        from.data[j] = tmp_view.data[k];
        from.shape.set_tensor_shape(j, tmp_view.tensor_shape(k));
        k++;
      } else {
        // directly from the input
        int src_idx = source_data_index_[i];
        from.data[j] = curr_batch.data[src_idx];
        from.shape.set_tensor_shape(j, curr_batch.tensor_shape(i));
      }
      to.data[j] = out_view[i].data;
      to.shape.set_tensor_shape(j, out_view.tensor_shape(i));
      j++;
    }
    kernels::KernelContext ctx;
    ctx.gpu.stream = ws.stream();
    kmgr_transpose_.Setup<TransposeKernel>(0, ctx, from.shape, make_span(perm), dtype.size());
    kmgr_transpose_.Run<TransposeKernel>(0, ctx, to, from);
  }
  for (int i = 0; i < nsamples; i++) {
    output.SetMeta(i, GetSample(i).get_meta());
  }
}

void NumpyReaderGPU::RunImpl(Workspace &ws) {
  auto &output = ws.Output<GPUBackend>(0);
  int ndim = output.shape().sample_dim();
  auto dtype = output.type();
  VALUE_SWITCH(ndim, Dims, NUMPY_ALLOWED_DIMS, (
    TYPE_SWITCH(dtype, type2id, T, NUMPY_ALLOWED_TYPES, (
      RunImplTyped<T, Dims>(ws);
    ), DALI_FAIL(make_string("Not supported input type: ", dtype)););  // NOLINT
  ), DALI_FAIL(make_string("Not supported number of dimensions: ", ndim)););  // NOLINT
}

}  // namespace dali
