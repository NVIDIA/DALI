// Copyright (c) 2017-2018, 2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef DALI_OPERATORS_READER_TFRECORD_READER_OP_H_
#define DALI_OPERATORS_READER_TFRECORD_READER_OP_H_

#ifdef DALI_BUILD_PROTO3

#include <utility>
#include "dali/operators/reader/loader/indexed_file_loader.h"
#include "dali/operators/reader/parser/tfrecord_parser.h"
#include "dali/operators/reader/reader_op.h"

namespace dali {

class TFRecordReader
    : public DataReader<CPUBackend, IndexedFileLoaderSample, Tensor<CPUBackend>, true> {
 public:
  explicit TFRecordReader(const OpSpec& spec)
  : DataReader<CPUBackend, IndexedFileLoaderSample, Tensor<CPUBackend>, true>(spec),
    dont_use_mmap_(spec.GetArgument<bool>("dont_use_mmap")),
    use_o_direct_(spec.GetArgument<bool>("use_o_direct")),
    thread_pool_(num_threads_, spec.GetArgument<int>("device_id"), false, "TFRecordReader") {
    DALI_ENFORCE(dont_use_mmap_  || !use_o_direct_, make_string("Cannot use use_o_direct with ",
                 "``dont_use_mmap=False``."));
    loader_ = InitLoader<IndexedFileLoader>(spec);
    parser_.reset(new TFRecordParser(spec));
    DALI_ENFORCE(!skip_cached_images_,
      "TFRecordReader doesn't support `skip_cached_images` option");
    this->SetInitialSnapshot();
  }

  void RunImpl(Workspace &ws) override {
    auto curr_batch_size = ws.NumInput() > 0 ? ws.GetInputBatchSize(0) : max_batch_size_;
    for (int i = 0; i < ws.NumOutput(); i++) {
      auto &output = ws.Output<CPUBackend>(i);
      output.SetSize(curr_batch_size);
    }
    auto &thread_pool = ws.GetThreadPool();
    for (int data_idx = 0; data_idx < curr_batch_size; ++data_idx) {
      auto job = [this, &ws, data_idx](int tid) {
        SampleWorkspace output_sample;
        MakeSampleView(output_sample, ws, data_idx, tid);
        const auto &loader_sample = GetSample(data_idx);
        if (loader_sample.work)
          loader_sample.work();
        parser_->Parse(loader_sample.tensor, &output_sample);
      };
      thread_pool.AddWork(std::move(job), -data_idx);  // -data_idx for FIFO order
    }
    // Run all tasks and wait for them to finish
    thread_pool.RunAll();
    // Propagate metadata from individual samples to the whole batch as working with SampleWorkspace
    // breaks metadata consistency - it sets it only to samples
    FixBatchPropertiesConsistency(ws, HasContiguousOutputs());
  }

  ~TFRecordReader() override {
    // Stop the prefetch thread as it uses the thread pool from this class. So before we can
    // destroy the thread pool make sure no one is using it anymore.
    this->StopPrefetchThread();
  }

  void Prefetch() override;

 protected:
  USE_READER_OPERATOR_MEMBERS(CPUBackend, IndexedFileLoaderSample, Tensor<CPUBackend>, true);
  bool dont_use_mmap_ = false;
  bool use_o_direct_ = false;
  size_t o_direct_chunk_size_ = 0;
  // ThreadPool for prefetch which is a separate thread
  ThreadPool thread_pool_;
};

}  // namespace dali

#endif  // DALI_BUILD_PROTO3
#endif  // DALI_OPERATORS_READER_TFRECORD_READER_OP_H_
