// Copyright (c) 2017-2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include <utility>
#include <vector>

#include "dali/operators/reader/loader/file_label_loader.h"
#include "dali/operators/reader/reader_op.h"

namespace dali {

class FileReader : public DataReader<CPUBackend, ImageLabelWrapper, ImageLabelWrapper, true> {
 public:
  explicit FileReader(const OpSpec& spec)
    : DataReader<CPUBackend, ImageLabelWrapper, ImageLabelWrapper, true>(spec) {
    bool shuffle_after_epoch = spec.GetArgument<bool>("shuffle_after_epoch");
    loader_ = InitLoader<FileLabelLoader>(spec, shuffle_after_epoch);
  }

  void RunImpl(SampleWorkspace &ws) override {
    const int idx = ws.data_idx();

    const auto& image_label = GetSample(idx);

    // copy from raw_data -> outputs directly
    auto &image_output = ws.Output<CPUBackend>(0);
    auto &label_output = ws.Output<CPUBackend>(1);

    Index image_size = image_label.image.size();

    image_output.Resize({image_size}, DALI_UINT8);
    label_output.Resize({1}, DALI_INT32);

    std::memcpy(image_output.raw_mutable_data(),
                image_label.image.raw_data(),
                image_size);
    image_output.SetSourceInfo(image_label.image.GetSourceInfo());

    label_output.mutable_data<int>()[0] = image_label.label;
  }

  void SaveState(OpCheckpoint &cpt, std::optional<cudaStream_t> stream) override {
    cpt.MutableCheckpointState() = loader_->PopStateSnapshot();
  }

  void RestoreState(const OpCheckpoint &cpt) override {
    loader_->RestoreStateFromSnapshot(cpt.CheckpointState<LoaderStateSnapshot>());
  }

  std::string SerializeCheckpoint(const OpCheckpoint &cpt) const override;

  void DeserializeCheckpoint(OpCheckpoint &cpt, const std::string &data) const override;

 protected:
  USE_READER_OPERATOR_MEMBERS(CPUBackend, ImageLabelWrapper, ImageLabelWrapper, true);
};

}  // namespace dali

#endif  // DALI_OPERATORS_READER_FILE_READER_OP_H_
