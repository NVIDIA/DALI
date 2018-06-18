// Copyright (c) 2017-2018, NVIDIA CORPORATION. All rights reserved.
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

#ifndef DALI_PIPELINE_OPERATORS_READER_FILE_READER_OP_H_
#define DALI_PIPELINE_OPERATORS_READER_FILE_READER_OP_H_

#include "dali/pipeline/operators/reader/reader_op.h"
#include "dali/pipeline/operators/reader/loader/file_loader.h"

namespace dali {

class FileReader : public DataReader<CPUBackend> {
 public:
  explicit FileReader(const OpSpec& spec)
    : DataReader<CPUBackend>(spec) {
    loader_.reset(new FileLoader(spec));
  }

  DEFAULT_READER_DESTRUCTOR(FileReader, CPUBackend);

  void RunImpl(SampleWorkspace *ws, const int i) override {
    const int idx = ws->data_idx();

    auto* raw_data = prefetched_batch_[idx];

    // copy from raw_data -> outputs directly
    auto *image_output = ws->Output<CPUBackend>(0);
    auto *label_output = ws->Output<CPUBackend>(1);

    Index raw_size = raw_data->size();
    Index image_size = raw_size - sizeof(int);

    image_output->Resize({image_size});
    image_output->mutable_data<uint8_t>();
    label_output->Resize({1});

    std::memcpy(image_output->raw_mutable_data(),
                raw_data->raw_data(),
                image_size);

    label_output->mutable_data<int>()[0] =
       *reinterpret_cast<const int*>(raw_data->data<uint8_t>() + image_size);
    return;
  }

 protected:
  USE_READER_OPERATOR_MEMBERS(CPUBackend);
};

}  // namespace dali

#endif  // DALI_PIPELINE_OPERATORS_READER_FILE_READER_OP_H_
