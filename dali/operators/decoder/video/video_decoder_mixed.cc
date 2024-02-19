// Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <vector>
#include "dali/pipeline/operator/operator.h"

namespace dali {

class VideoDecoderMixed : public Operator<MixedBackend> {
 public:
  explicit VideoDecoderMixed(const OpSpec &spec): Operator<MixedBackend>(spec) {
    DALI_FAIL(
        "Video operators are now part of a separate package. "
        "Please use the operators from module `nvidia.dali.fn.plugin.video` instead");
  }

  bool CanInferOutputs() const override { return true; }
  void Run(Workspace &ws) override { assert(false); }
  bool SetupImpl(std::vector<OutputDesc> &output_desc, const Workspace &ws) override { assert(false); }
};

DALI_REGISTER_OPERATOR(experimental__decoders__Video, VideoDecoderMixed, Mixed);

}  // namespace dali
