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

#ifndef DALI_OPERATORS_READER_NUMPY_READER_OP_H_
#define DALI_OPERATORS_READER_NUMPY_READER_OP_H_

#include <utility>
#include <string>
#include <vector>
#include "dali/operators/reader/reader_op.h"
#include "dali/operators/reader/loader/numpy_loader.h"

namespace dali {

class NumpyReader : public DataReader<CPUBackend, ImageFileWrapper > {
 public:
  explicit NumpyReader(const OpSpec& spec)
    : DataReader< CPUBackend, ImageFileWrapper >(spec) {
    bool shuffle_after_epoch = spec.GetArgument<bool>("shuffle_after_epoch");
    loader_ = InitLoader<NumpyLoader>(spec, std::vector<string>(),
                                      shuffle_after_epoch);
  }

  void RunImpl(SampleWorkspace &ws) override {
    const int idx = ws.data_idx();

    const auto& imfile = GetSample(idx);

    // copy from raw_data -> outputs directly
    auto &image_output = ws.Output<CPUBackend>(0);

    // image
    Index image_bytes = imfile.image.nbytes();
    image_output.Resize(imfile.image.shape(), imfile.image.type());

    std::memcpy(image_output.raw_mutable_data(),
                imfile.image.raw_data(),
                image_bytes);
    image_output.SetSourceInfo(imfile.image.GetSourceInfo());
  }

 protected:
  USE_READER_OPERATOR_MEMBERS(CPUBackend, ImageFileWrapper);
};

}  // namespace dali

#endif  // DALI_OPERATORS_READER_NUMPY_READER_OP_H_
