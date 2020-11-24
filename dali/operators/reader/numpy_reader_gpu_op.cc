// Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
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

#include "dali/pipeline/data/views.h"
#include "dali/kernels/transpose/transpose_gpu.h"
#include "dali/core/static_switch.h"
#include "dali/operators/reader/numpy_reader_gpu_op.h"

namespace dali {

void NumpyReaderGPU::Prefetch() {
  // We actually prepare the next batch
  TimeRange tr("NumpyReaderGPU::Prefetch #" + to_string(curr_batch_producer_), TimeRange::kRed);
  DataReader<GPUBackend, ImageFileWrapperGPU>::Prefetch();
  auto &curr_batch = prefetched_batch_queue_[curr_batch_producer_];
  auto &curr_tensor_list = prefetched_batch_tensors_[curr_batch_producer_];

  // get shapes
  for (size_t data_idx = 0; data_idx < curr_batch.size(); ++data_idx) {
    thread_pool_.AddWork([&curr_batch, data_idx](int tid) {
        curr_batch[data_idx]->read_meta_f();
      });
  }
  thread_pool_.RunAll();

  // resize the current batch
  std::vector<TensorShape<>> tmp_shapes;
  auto ref_type = curr_batch[0]->type_info;
  auto ref_shape = curr_batch[0]->shape;
  for (size_t data_idx = 0; data_idx < curr_batch.size(); ++data_idx) {
    auto &sample = curr_batch[data_idx];
    DALI_ENFORCE(ref_type == sample->type_info, make_string("Inconsistent data! "
                 "The data produced by the reader has inconsistent type:\n"
                 "type of [", data_idx, "] is ", sample->type_info.id(), " whereas\n"
                 "type of [0] is ", ref_type.id()));

    DALI_ENFORCE(ref_shape.size() == sample->shape.size(), make_string("Inconsistent data! "
        "The data produced by the reader has inconsistent dimensionality:\n"
        "[", data_idx, "] has ", sample->shape.size(), " dimensions whereas\n"
        "[0] has ", ref_shape.size(), " dimensions."));
    DALI_ENFORCE(ref_shape.size() == sample->shape.size(),
                  make_string("The tensors in the input batch do not all have the same number of "
                  "dimensions for sample [0]: ", ref_shape.size(), " vs [", data_idx, "]: ",
                  sample->shape.size(), "."));
    tmp_shapes.push_back(sample->shape);
  }

  curr_tensor_list.Resize(TensorListShape<>(tmp_shapes), ref_type);

  // read the data
  for (size_t data_idx = 0; data_idx < curr_tensor_list.ntensor(); ++data_idx) {
    thread_pool_.AddWork([&curr_batch, &curr_tensor_list, data_idx](int tid) {
        curr_batch[data_idx]->read_sample_f(curr_tensor_list.raw_mutable_data(),
                                            curr_tensor_list.tensor_offset(data_idx) *
                                            curr_tensor_list.type().size(),
                                            curr_tensor_list.nbytes());
      });
  }
  thread_pool_.RunAll();
}

void PermuteHelper(const TensorShape<> &plain_shapes, std::vector<int64_t> &perm_shape,
                  std::vector<int> &perm) {
  int n_dims = plain_shapes.size();
  if (perm.empty()) {
    perm.resize(n_dims);
    for (int i = 0; i < n_dims; ++i) {
      perm[i] = n_dims - i - 1;
    }
  }
  for (int i = 0; i < n_dims; ++i) {
    perm_shape[i] = plain_shapes[perm[i]];
  }
}

void NumpyReaderGPU::RunImpl(DeviceWorkspace &ws) {
  TensorListShape<> shape(batch_size_);
  // use vector for temporarily storing shapes
  std::vector<TensorShape<>> tmp_shapes;
  std::vector<TensorShape<>> transpose_shapes;
  std::vector<int> perm;
  std::vector<int64_t> perm_shape;

  perm.reserve(GetSampleShape(0).size());
  perm_shape.resize(GetSampleShape(0).size());

  for (int sample_idx = 0; sample_idx < batch_size_; sample_idx++) {
    const auto& imfile = GetSample(sample_idx);
    auto plain_shape = GetSampleShape(sample_idx);
    if (imfile.meta == "transpose:false") {
      tmp_shapes.push_back(plain_shape);
    } else {
      PermuteHelper(plain_shape, perm_shape, perm);
      tmp_shapes.push_back(perm_shape);
      transpose_shapes.push_back(plain_shape);
    }
  }
  auto ref_type = GetSampleType(0);
  shape = TensorListShape<>(tmp_shapes);
  ws.Output<GPUBackend>(0).Resize(shape, ref_type);

  auto &image_output = ws.Output<GPUBackend>(0);

  SmallVector<int64_t, 256> copy_sizes;
  copy_sizes.reserve(batch_size_);
  SmallVector<const void *, 256> copy_from;
  copy_from.reserve(batch_size_);
  SmallVector<void *, 256> copy_to;
  copy_to.reserve(batch_size_);

  SmallVector<const void *, 256> transpose_from;
  transpose_from.reserve(batch_size_);
  SmallVector<void *, 256> transpose_to;
  transpose_to.reserve(batch_size_);


  for (int data_idx = 0; data_idx < batch_size_; ++data_idx) {
    const auto& imfile = GetSample(data_idx);
    if (imfile.meta == "transpose:false") {
      copy_from.push_back(GetSampleRawData(data_idx));
      copy_to.push_back(image_output.raw_mutable_tensor(data_idx));
      copy_sizes.push_back(shape.tensor_size(data_idx));
    } else {
      transpose_from.push_back(GetSampleRawData(data_idx));
      transpose_to.push_back(image_output.raw_mutable_tensor(data_idx));
    }
    image_output.SetSourceInfo(data_idx, imfile.image.GetSourceInfo());
  }

  // use copy kernel for plan samples
  if (!copy_sizes.empty()) {
    ref_type.template Copy<GPUBackend, GPUBackend>(copy_to.data(), copy_from.data(),
                                                   copy_sizes.data(), copy_sizes.size(),
                                                   ws.stream(), true);
  }

  // transpose remaining samples
  if (!transpose_from.empty()) {
    kernels::KernelContext ctx;
    ctx.gpu.stream = ws.stream();
    kmgr_.Setup<TransposeKernel>(0, ctx, TensorListShape<>(transpose_shapes), make_span(perm),
                                 ref_type.size());
    kmgr_.Run<TransposeKernel>(0, 0, ctx, transpose_to.data(), transpose_from.data());
  }
}

DALI_REGISTER_OPERATOR(NumpyReader, NumpyReaderGPU, GPU);

}  // namespace dali
