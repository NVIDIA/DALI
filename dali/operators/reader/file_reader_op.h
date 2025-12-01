// Copyright (c) 2017-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef DALI_OPERATORS_READER_FILE_READER_OP_H_
#define DALI_OPERATORS_READER_FILE_READER_OP_H_

#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "dali/core/small_vector.h"
#include "dali/operators/reader/loader/file_label_loader.h"
#include "dali/operators/reader/reader_op.h"

namespace dali {

class FileReader : public DataReader<CPUBackend, ImageLabelWrapper, ImageLabelWrapper, true> {
 public:
  explicit FileReader(const OpSpec& spec)
    : DataReader<CPUBackend, ImageLabelWrapper, ImageLabelWrapper, true>(spec) {
    bool shuffle_after_epoch = spec.GetArgument<bool>("shuffle_after_epoch");
    loader_ = InitLoader<FileLabelLoader>(spec, shuffle_after_epoch);
    this->SetInitialSnapshot();
  }

  bool HasContiguousOutputs() const override {
    return true;
  }

  bool SetupImpl(std::vector<OutputDesc>& output_desc, const Workspace& ws) override {
    // If necessary start prefetching thread and wait for a consumable batch
    DataReader<CPUBackend, ImageLabelWrapper, ImageLabelWrapper, true>::SetupImpl(output_desc, ws);
    auto samples = GetCurrBatch();
    int batch_size = samples.size();

    output_desc.resize(2);
    output_desc[0].shape.resize(batch_size, 1);
    output_desc[0].type = DALI_UINT8;
    output_desc[1].shape = uniform_list_shape<1>(batch_size, {1});
    output_desc[1].type = DALI_INT32;

    TensorListShape<1> out_shape(batch_size);
    for (int sample_idx = 0; sample_idx < batch_size; ++sample_idx) {
      auto& sample = *samples[sample_idx];
      int64_t image_size =
          sample.file_stream != nullptr ? sample.file_stream->Size() : sample.image.size();
      output_desc[0].shape.tensor_shape_span(sample_idx)[0] = image_size;
      output_desc[1].shape.tensor_shape_span(sample_idx)[0] = 1;
    }
    return true;
  }

  void RunImpl(Workspace &ws) override {
    auto &file_output = ws.Output<CPUBackend>(0);
    auto &label_output = ws.Output<CPUBackend>(1);
    auto samples = GetCurrBatch();
    int batch_size = samples.size();

    auto &thread_pool = ws.GetThreadPool();
    std::unordered_map<void *, SmallVector<int, 4>> unique_samples;
    for (int sample_idx = 0; sample_idx < batch_size; ++sample_idx) {
      auto sample_ptr = samples[sample_idx].get();
      unique_samples[sample_ptr].push_back(sample_idx);
    }
    for (auto &sample : unique_samples) {
      auto &sample_idxs = sample.second;
      assert(!sample_idxs.empty());
      thread_pool.AddWork([&, sample_idxs = std::move(sample_idxs)](int tid) {
        int sample_idx = sample_idxs[0];
        // Read the first sample (the next are repeated)
        auto &sample = *samples[sample_idx];
        if (sample.file_stream != nullptr) {
          sample.file_stream->SeekRead(0, SEEK_SET);
          int64_t sample_sz = sample.file_stream->Size();
          int64_t read_nbytes =
              sample.file_stream->Read(file_output.raw_mutable_tensor(sample_idx), sample_sz);
          sample.file_stream->Close();
          DALI_ENFORCE(read_nbytes == sample_sz,
                       make_string("Failed to read file: ", sample.file_stream->path()));
          sample.file_stream.reset();
        } else {
          std::memcpy(file_output.raw_mutable_tensor(sample_idx), sample.image.raw_data(),
                      sample.image.size());
        }
        file_output.SetSourceInfo(sample_idx, sample.image.GetSourceInfo());
        label_output.mutable_tensor<int>(sample_idx)[0] = sample.label;

        // Now copy the sample we read to any repeated samples
        for (size_t i = 1; i < sample_idxs.size(); i++) {
          int repeated_sample_idx = sample_idxs[i];
          std::memcpy(file_output.raw_mutable_tensor(repeated_sample_idx),
                      file_output.raw_mutable_tensor(sample_idx),
                      file_output.shape().tensor_size(sample_idx));
          file_output.SetSourceInfo(repeated_sample_idx,
                                    file_output.GetMeta(sample_idx).GetSourceInfo());
          label_output.mutable_tensor<int>(repeated_sample_idx)[0] =
              label_output.mutable_tensor<int>(sample_idx)[0];
        }
      }, file_output.shape().tensor_size(sample_idxs[0]));
    }
    thread_pool.RunAll();
  }

 protected:
  USE_READER_OPERATOR_MEMBERS(CPUBackend, ImageLabelWrapper, ImageLabelWrapper, true);
};

}  // namespace dali

#endif  // DALI_OPERATORS_READER_FILE_READER_OP_H_
