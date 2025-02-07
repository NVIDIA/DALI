// Copyright (c) 2023-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "dali/operators/input/video_input.h"
#include <memory>

namespace dali {

class VideoInputMixed : public VideoInput<MixedBackend> {
  /*
   * This awkward class originates from an API inconsistency between
   * Operator<CPUBackend> and Operator<MixedBackend>. Operator<CPUBackend> has a `RunImpl` function
   * to be overriden, while Operator<MixedBackend> has `Run` function to be overriden.
   * Can't sort it out using SFINAE, since these are virtual functions.
   */
 public:
  explicit VideoInputMixed(const OpSpec &spec) : VideoInput<MixedBackend>(spec) {}
  void RunImpl(Workspace &ws) override { VideoInputRunImpl(ws); }
};

DALI_REGISTER_OPERATOR(experimental__inputs__Video, VideoInputMixed, Mixed);

}  // namespace dali
