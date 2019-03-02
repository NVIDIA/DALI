// Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
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

#ifndef DALI_PIPELINE_OPERATORS_DECODER_NVJPEG_DECODER_SPLITTED_H_
#define DALI_PIPELINE_OPERATORS_DECODER_NVJPEG_DECODER_SPLITTED_H_

#include <nvjpeg.h>
#include <functional>
#include <string>
#include <utility>
#include <vector>

#include "dali/pipeline/operators/operator.h"

namespace dali {

class nvJPEGDecoderSplitted : public Operator<MixedBackend> {
 public:
  explicit nvJPEGDecoderSplitted(const OpSpec& spec) :
    Operator<MixedBackend>(spec) {
      // Dummy class: we only need the schema
      // TODO(spanev): replace by nvJPEGDecoderNew
  }
  using dali::OperatorBase::Run;
  void Run(MixedWorkspace *ws) override {
    // Nothing to see here
  }
};

}  // namespace dali

#endif  // DALI_PIPELINE_OPERATORS_DECODER_NVJPEG_DECODER_SPLITTED_H_
