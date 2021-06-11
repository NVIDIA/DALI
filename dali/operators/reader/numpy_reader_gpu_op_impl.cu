// Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include "dali/kernels/slice/slice_flip_normalize_permute_pad_common.h"
#include "dali/kernels/slice/slice_flip_normalize_permute_pad_gpu.h"
#include "dali/kernels/slice/slice_gpu.cuh"
#include "dali/kernels/slice/slice_kernel_utils.h"
#include "dali/kernels/transpose/transpose_gpu.h"
#include "dali/operators/reader/numpy_reader_gpu_op.h"
#include "dali/pipeline/data/views.h"

namespace dali {

template <typename T, int Dims>
void NumpyReaderGPU::RunImplTyped(DeviceWorkspace &ws) {
  auto &output = ws.OutputRef<GPUBackend>(0);
  const auto &out_sh = output.shape();
  int ndim = out_sh.sample_dim();
  int nsamples = out_sh.num_samples();
  auto dtype = GetSampleType(0);
  bool need_slice = !rois_.empty();
  auto out_view = view<T>(output);

  // Permuted dims, to use for the transposition
  std::array<int, Dims> perm;
  for (int d = 0; d < Dims; d++)
    perm[d] = Dims - 1 - d;

  for (int data_idx = 0; data_idx < nsamples; data_idx++) {
    const auto& imfile = GetSample(data_idx);
    output.SetSourceInfo(data_idx, imfile.image.GetSourceInfo());
  }

  int nsamples_copy = need_copy_.size();
  if (nsamples_copy) {
    if (nsamples_copy == nsamples) {
      std::swap(output, prefetched_batch_tensors_[curr_batch_consumer_]);
    } else {
      auto type_sz = dtype.size();
      for (int i : need_copy_) {
        const auto& file_i = GetSample(i);
        auto sz = out_sh.tensor_size(i) * type_sz;
        sg_.AddCopy(out_view.data[i], GetSampleRawData(i), sz);
      }
      sg_.Run(ws.stream());
    }
  }

  int nsamples_slice = need_slice_.size();
  if (nsamples_slice) {
    // TLV used to invoke kernels with a subset of the samples
    TensorListView<StorageGPU, const T, Dims> from;
    from.shape.resize(nsamples_slice);
    from.data.resize(nsamples_slice);
    TensorListView<StorageGPU, T, Dims> to;
    to.shape.resize(nsamples_slice);
    to.data.resize(nsamples_slice);

    std::vector<kernels::SliceArgs<T, Dims>> slice_args;
    slice_args.resize(nsamples_slice);
    int j = 0;
    for (int i : need_slice_) {
      const auto& file_i = GetSample(i);
      auto &args = slice_args[j];
      args.anchor = rois_[i].anchor;
      args.shape = rois_[i].shape;
      args.fill_values.clear();
      args.fill_values.push_back(ConvertSat<T>(fill_value_));
      from.data[j] = GetSampleData<T>(i);
      from.shape.set_tensor_shape(j, GetSampleShape(i).template to_static<Dims>());
      to.data[j] = out_view.data[i];
      to.shape.set_tensor_shape(j, out_view.shape.tensor_shape(i).template to_static<Dims>());
      j++;
    }
    kernels::KernelContext ctx;
    ctx.gpu.stream = ws.stream();
    using Kernel = kernels::SliceGPU<T, T, Dims>;
    kmgr_slice_.Resize<Kernel>(1, 1);
    kmgr_slice_.Setup<Kernel>(0, ctx, from, slice_args);
    kmgr_slice_.Run<Kernel>(0, 0, ctx, to, from, slice_args);
  }

  int nsamples_slice_perm = need_slice_perm_.size();
  if (nsamples_slice_perm) {
    // TLV used to invoke kernels with a subset of the samples
    TensorListView<StorageGPU, const T, Dims> from;
    from.shape.resize(nsamples_slice_perm);
    from.data.resize(nsamples_slice_perm);
    TensorListView<StorageGPU, T, Dims> to;
    to.shape.resize(nsamples_slice_perm);
    to.data.resize(nsamples_slice_perm);

    using Args = kernels::SliceFlipNormalizePermutePadArgs<Dims>;
    std::vector<Args> slice_transpose_args;
    slice_transpose_args.resize(nsamples_slice_perm);
    int j = 0;
    for (int i : need_slice_perm_) {
      const auto& file_i = GetSample(i);
      auto &args = slice_transpose_args[j];
      args = Args(rois_[i].shape, GetSampleShape(i));
      args.anchor = rois_[i].anchor;
      for (int d = 0; d < Dims; d++)
        args.permuted_dims[d] = Dims - 1 - d;
      args.fill_values.clear();
      args.fill_values.push_back(ConvertSat<T>(fill_value_));
      from.data[j] = GetSampleData<T>(i);
      from.shape.set_tensor_shape(j, GetSampleShape(i).template to_static<Dims>());
      to.data[j] = out_view.data[i];
      to.shape.set_tensor_shape(j, out_view.shape.tensor_shape(i).template to_static<Dims>());
      j++;
    }
    kernels::KernelContext ctx;
    ctx.gpu.stream = ws.stream();
    using Kernel = kernels::SliceFlipNormalizePermutePadGpu<T, T, Dims>;
    kmgr_slice_perm_.Resize<Kernel>(1, 1);
    kmgr_slice_perm_.Setup<Kernel>(0, ctx, from, slice_transpose_args);
    kmgr_slice_perm_.Run<Kernel>(0, 0, ctx, to, from, slice_transpose_args);
  }

  int nsamples_transpose = need_transpose_.size();
  if (nsamples_transpose) {
    // TLV used to invoke kernels with a subset of the samples
    TensorListView<StorageGPU, const T> from;
    TensorListView<StorageGPU, T> to;
    from.shape.resize(nsamples_transpose, ndim);
    from.data.resize(nsamples_transpose);
    to.shape.resize(nsamples_transpose, ndim);
    to.data.resize(nsamples_transpose);

    int j = 0;
    for (int i : need_transpose_) {
      const auto& file_i = GetSample(i);
      from.data[j] = GetSampleData<T>(i);
      from.shape.set_tensor_shape(j, GetSampleShape(i));
      to.data[j] = out_view.data[i];
      to.shape.set_tensor_shape(j, out_view.shape.tensor_shape(i));
      j++;
    }
    kernels::KernelContext ctx;
    ctx.gpu.stream = ws.stream();
    kmgr_transpose_.Setup<TransposeKernel>(0, ctx, from.shape, make_span(perm), dtype.size());
    kmgr_transpose_.Run<TransposeKernel>(0, 0, ctx, to, from);
  }
}

void NumpyReaderGPU::RunImpl(DeviceWorkspace &ws) {
  auto &output = ws.OutputRef<GPUBackend>(0);
  int ndim = output.shape().sample_dim();
  auto dtype = output.type().id();
  VALUE_SWITCH(ndim, Dims, NUMPY_ALLOWED_DIMS, (
    TYPE_SWITCH(dtype, type2id, T, NUMPY_ALLOWED_TYPES, (
      RunImplTyped<T, Dims>(ws);
    ), DALI_FAIL(make_string("Not supported input type: ", dtype)););  // NOLINT
  ), DALI_FAIL(make_string("Not supported number of dimensions: ", ndim)););  // NOLINT
}

}  // namespace dali
