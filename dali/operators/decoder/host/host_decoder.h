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

#ifndef DALI_OPERATORS_DECODER_HOST_HOST_DECODER_H_
#define DALI_OPERATORS_DECODER_HOST_HOST_DECODER_H_

#include <vector>

#include "dali/core/common.h"
#include "dali/core/error_handling.h"
#include "dali/pipeline/operator/operator.h"
#include "dali/util/crop_window.h"

namespace dali {

class HostDecoder : public Operator<CPUBackend> {
 public:
  explicit inline HostDecoder(const OpSpec &spec) :
      Operator<CPUBackend>(spec),
      output_type_(spec.GetArgument<DALIImageType>("output_type")),
      use_fast_idct_(spec.GetArgument<bool>("use_fast_idct"))
  {}

  inline ~HostDecoder() override = default;
  DISABLE_COPY_MOVE_ASSIGN(HostDecoder);

 protected:
  bool SetupImpl(std::vector<OutputDesc> &output_desc, const HostWorkspace &ws) override {
    return false;
  }

  void RunImpl(SampleWorkspace &ws) override;

  virtual CropWindowGenerator GetCropWindowGenerator(int data_idx) const {
    return {};
  }

  DALIImageType output_type_;
  bool use_fast_idct_ = false;
};

}  // namespace dali

#endif  // DALI_OPERATORS_DECODER_HOST_HOST_DECODER_H_
