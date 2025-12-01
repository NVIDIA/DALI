// Copyright (c) 2020-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include "dali/kernels/common/copy.h"

namespace dali {

NumpyReaderGPU::NumpyReaderGPU(const OpSpec& spec)
    : NumpyReader<GPUBackend, NumpyFileWrapperGPU>(spec),
      thread_pool_(num_threads_, spec.GetArgument<int>("device_id"), false, "NumpyReaderGPU"),
      sg_(1 << 18),
      header_cache_(spec.GetArgument<bool>("cache_header_information")) {
  InitDriverScope();
  prefetched_batch_tensors_.resize(prefetch_queue_depth_);
  // make the device current
  DeviceGuard g(device_id_);

  staging_stream_ = CUDAStreamPool::instance().Get();
  staging_ready_ = CUDAEventPool::instance().Get();
  staging_.set_stream(staging_stream_);

  // init loader
  bool shuffle_after_epoch = spec.GetArgument<bool>("shuffle_after_epoch");
  loader_ = InitLoader<NumpyLoaderGPU>(spec, shuffle_after_epoch);
  this->SetInitialSnapshot();

  kmgr_transpose_.Resize<TransposeKernel>(1);
}

NumpyReaderGPU::~NumpyReaderGPU() {
  // Stop the prefetch thread as it uses the thread pool from this class. So before we can
  // destroy the thread pool make sure no one is using it anymore.
  this->StopPrefetchThread();
}

void NumpyReaderGPU::Prefetch() {
  // We actually prepare the next batch
  DomainTimeRange tr("[DALI][NumpyReaderGPU] Prefetch #" + to_string(curr_batch_producer_),
                      DomainTimeRange::kRed);
  DataReader<GPUBackend, NumpyFileWrapperGPU, NumpyFileWrapperGPU, true>::Prefetch();
  auto &curr_batch = prefetched_batch_queue_[curr_batch_producer_];
  auto &curr_tensor_list = prefetched_batch_tensors_[curr_batch_producer_];

  // get shapes
  for (size_t data_idx = 0; data_idx < curr_batch.size(); ++data_idx) {
    // when padding, the last sample is duplicated so no need to redo the same work
    if (data_idx > 0 && curr_batch[data_idx -1 ] == curr_batch[data_idx]) continue;
    thread_pool_.AddWork([this, &curr_batch, data_idx](int tid) {
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
  int first_padded = -1;
  for (int data_idx = 0; data_idx < curr_tensor_list.num_samples(); ++data_idx) {
    curr_tensor_list.SetMeta(data_idx, curr_batch[data_idx]->meta);
    SampleView<GPUBackend> sample(curr_tensor_list.raw_mutable_tensor(data_idx),
                                  curr_tensor_list.tensor_shape(data_idx),
                                  curr_tensor_list.type());
    // when padding, the last sample is duplicated so no need to redo the same work
    if (data_idx > 0 && curr_batch[data_idx - 1] == curr_batch[data_idx]) {
      if (first_padded < 0) {
        first_padded = data_idx;
      }
      curr_batch[data_idx]->source_sample_idx = first_padded - 1;
    } else {
      ScheduleChunkedRead(sample, *curr_batch[data_idx]);
      curr_batch[data_idx]->source_sample_idx = data_idx;
    }
  }
  thread_pool_.RunAll();
  staging_.commit();
  CUDA_CALL(cudaEventRecord(staging_ready_, staging_stream_));

  for (int data_idx = 0; data_idx < curr_tensor_list.num_samples(); ++data_idx) {
    curr_batch[data_idx]->file_stream_->Close();
  }
}

void NumpyReaderGPU::ScheduleChunkedRead(SampleView<GPUBackend> &out_sample,
                                         NumpyFileWrapperGPU &load_target) {
  // TODO(michalz): add nbytes and num_elements to SampleView.
  size_t data_bytes = out_sample.shape().num_elements() *
                      TypeTable::GetTypeInfo(out_sample.type()).size();
  if (!data_bytes)
    return;  // empty array - short-circuit

  uint8_t *base_ptr = static_cast<uint8_t*>(out_sample.raw_mutable_data());
  uint8_t *dst_ptr = base_ptr;
  ssize_t read_start = load_target.data_offset & -gds::kGDSAlignment;  // align _down_
  ssize_t file_offset = read_start;
  ssize_t read_bytes = data_bytes + load_target.data_offset - read_start;
  while (read_bytes > 0) {
    ssize_t chunk_read_length = std::min<ssize_t>(read_bytes, chunk_size_);
    ssize_t copy_start = std::max(file_offset, load_target.data_offset);
    ssize_t copy_skip = copy_start - file_offset;
    ssize_t copy_end = file_offset + chunk_read_length;
    ssize_t chunk_copy_length = copy_end - copy_start;
    thread_pool_.AddWork([=, &load_target, this](int tid) {
      assert(chunk_read_length <= static_cast<ssize_t>(staging_.chunk_size()));
      auto buffer = staging_.get_staging_buffer();
      load_target.ReadRawChunk(buffer.at(0), chunk_read_length, 0, file_offset);
      assert(dst_ptr >= base_ptr && dst_ptr + chunk_copy_length <= base_ptr + data_bytes);
      staging_.copy_to_client(dst_ptr, chunk_copy_length, std::move(buffer), copy_skip);
    });

    // update addresses
    dst_ptr += chunk_copy_length;
    file_offset += chunk_read_length;
    read_bytes -= chunk_read_length;
  }
  assert(dst_ptr == base_ptr + data_bytes);
}

DALI_REGISTER_OPERATOR(readers__Numpy, NumpyReaderGPU, GPU);

// Deprecated alias
DALI_REGISTER_OPERATOR(NumpyReader, NumpyReaderGPU, GPU);

}  // namespace dali
