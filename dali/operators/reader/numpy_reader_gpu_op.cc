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

#include <string>
#include "dali/operators/reader/numpy_reader_gpu_op.h"
#include "dali/pipeline/data/views.h"

namespace dali {

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
        curr_batch[data_idx]->read_meta_f();
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
                 "type of [", data_idx, "] is ", sample->get_type().id(), " whereas\n"
                 "type of [0] is ", ref_type.id()));

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

  size_t chunk_size = static_cast<size_t>( \
                        div_ceil(static_cast<uint64_t>(curr_tensor_list.nbytes()),
                                 static_cast<uint64_t>(thread_pool_.NumThreads())));

  // read the data
  for (size_t data_idx = 0; data_idx < curr_tensor_list.ntensor(); ++data_idx) {
    curr_tensor_list.SetMeta(data_idx, curr_batch[data_idx]->get_meta());
    size_t image_bytes = static_cast<size_t>(volume(curr_tensor_list.tensor_shape(data_idx))
                                             * curr_tensor_list.type().size());
    uint8_t* dst_ptr = static_cast<uint8_t*>(curr_tensor_list.raw_mutable_tensor(data_idx));
    size_t file_offset = 0;
    while (image_bytes > 0) {
      size_t read_bytes = std::min(image_bytes, chunk_size);
      void* buffer = static_cast<void*>(dst_ptr);
      thread_pool_.AddWork([&curr_batch, data_idx, buffer, file_offset, read_bytes](int tid) {
        curr_batch[data_idx]->read_sample_f(buffer, file_offset, read_bytes);
      });

      // update addresses
      dst_ptr += read_bytes;
      file_offset += read_bytes;
      image_bytes -= read_bytes;
    }
  }
  thread_pool_.RunAll();

  for (size_t data_idx = 0; data_idx < curr_tensor_list.ntensor(); ++data_idx) {
    curr_batch[data_idx]->file_stream->Close();
  }
}

DALI_REGISTER_OPERATOR(readers__Numpy, NumpyReaderGPU, GPU);

// Deprecated alias
DALI_REGISTER_OPERATOR(NumpyReader, NumpyReaderGPU, GPU);

}  // namespace dali
