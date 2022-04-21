// Copyright (c) 2020-2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <memory>
#include <string>
#include "dali/core/mm/memory.h"
#include "dali/core/mm/malloc_resource.h"
#include "dali/operators/reader/numpy_reader_gpu_op.h"
#include "dali/operators/reader/gds_mem.h"
#include "dali/pipeline/data/views.h"

namespace dali {

NumpyReaderGPU::NumpyReaderGPU(const OpSpec& spec)
    : NumpyReader<GPUBackend, NumpyFileWrapperGPU>(spec),
      thread_pool_(num_threads_, spec.GetArgument<int>("device_id"), false),
      sg_(1 << 18),
      header_cache_(spec.GetArgument<bool>("cache_header_information")) {
  prefetched_batch_tensors_.resize(prefetch_queue_depth_);
  // make the device current
  DeviceGuard g(device_id_);

  // init loader
  bool shuffle_after_epoch = spec.GetArgument<bool>("shuffle_after_epoch");
  loader_ = InitLoader<NumpyLoaderGPU>(spec, std::vector<string>(), shuffle_after_epoch);

  kmgr_transpose_.Resize<TransposeKernel>(1);
}

void NumpyReaderGPU::Prefetch() {
  // We actually prepare the next batch
  DomainTimeRange tr("[DALI][NumpyReaderGPU] Prefetch #" + to_string(curr_batch_producer_),
                      DomainTimeRange::kRed);
  DataReader<GPUBackend, NumpyFileWrapperGPU>::Prefetch();
  auto &curr_batch = prefetched_batch_queue_[curr_batch_producer_];
  auto &curr_tensor_list = prefetched_batch_tensors_[curr_batch_producer_];

  // get shapes
  for (size_t data_idx = 0; data_idx < curr_batch.size(); ++data_idx) {
    thread_pool_.AddWork([&curr_batch, data_idx](int tid) {
        curr_batch[data_idx]->Reopen();
        curr_batch[data_idx]->ReadHeader(header_cache_);
      });
  }
  thread_pool_.RunAll();

  // resize the current batch
  auto ref_type = curr_batch[0]->get_type();
  auto ref_shape = curr_batch[0]->get_shape();
  TensorListShape<> tmp_shapes(curr_batch.size(), ref_shape.sample_dim());
  for (size_t data_idx = 0; data_idx < curr_batch.size(); ++data_idx) {
    auto &sample = curr_batch[data_idx];
    DALI_ENFORCE(ref_type == sample->get_type(), make_string("Inconsistent data! "
                 "The data produced by the reader has inconsistent type:\n"
                 "type of [", data_idx, "] is ", sample->get_type(), " whereas\n"
                 "type of [0] is ", ref_type));

    DALI_ENFORCE(
        ref_shape.sample_dim() == sample->get_shape().sample_dim(),
        make_string(
            "Inconsistent data! The data produced by the reader has inconsistent dimensionality:\n"
            "[",
            data_idx, "] has ", sample->get_shape().sample_dim(),
            " dimensions whereas\n"
            "[0] has ",
            ref_shape.sample_dim(), " dimensions."));
    tmp_shapes.set_tensor_shape(data_idx, sample->get_shape());
  }

  curr_tensor_list.Resize(tmp_shapes, ref_type);

  // read the data
  for (int data_idx = 0; data_idx < curr_tensor_list.num_samples(); ++data_idx) {
    curr_tensor_list.SetMeta(data_idx, curr_batch[data_idx]->meta);
    ChunkedRead(curr_tensor_list[data_idx], curr_batch[data_idx]);
  }
  thread_pool_.RunAll();

  for (int data_idx = 0; data_idx < curr_tensor_list.num_samples(); ++data_idx) {
    curr_batch[data_idx]->file_stream->Close();
  }
}

void NumpyReaderGPU::ChunkedRead(const SampleView<GPUBackend> &out_sample,
                                 NumpyFileWrapperGPU &load_target) {
  // TODO(michalz): add nbytes and num_elements to SampleView.
  size_t data_bytes = out_sample.shape().num_elements() *
                      TypeTable::GetTypeInfo(out_sample.type()).size();

  uint8_t* dst_ptr = static_cast<uint8_t*>(out_sample.raw_mutable_data());
  ssize_t read_start = load_target.data_offset & -kGDSAlignment;  // align _down_
  ssize_t file_offset = ;
  ssize_t read_bytes = data_bytes + (target.data_offset - read_start);
  while (read_bytes > 0) {
    size_t this_chunk = std::min(read_bytes, chunk_size_);
    thread_pool_.AddWork([this, &load_target, dst_ptr, file_offset, this_chunk](int tid) {
      void *chunk = staging_.GetChunk();
      load_target->ReadChunk(buffer, file_offset, read_bytes);
    });

    // update addresses
    dst_ptr += read_bytes;
    file_offset += read_bytes;
    image_bytes -= read_bytes;
  }
}

DALI_REGISTER_OPERATOR(readers__Numpy, NumpyReaderGPU, GPU);

// Deprecated alias
DALI_REGISTER_OPERATOR(NumpyReader, NumpyReaderGPU, GPU);

}  // namespace dali
