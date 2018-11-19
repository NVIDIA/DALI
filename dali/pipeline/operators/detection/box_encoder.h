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

#ifndef DALI_PIPELINE_OPERATORS_DETECTION_BOX_ENCODER_H_
#define DALI_PIPELINE_OPERATORS_DETECTION_BOX_ENCODER_H_

#include <cstring>
#include <vector>

#include "dali/pipeline/operators/operator.h"

namespace dali {

template <typename Backend>
class SSDBoxEncoder : public Operator<Backend> {
 public:
  explicit SSDBoxEncoder(const OpSpec &spec)
      : Operator<Backend>(spec), criteria_(spec.GetArgument<float>("criteria")) {
    DALI_ENFORCE(criteria_ >= 0.f);
    DALI_ENFORCE(criteria_ <= 1.f);

    auto anchors = spec.GetArgument<vector<float>>("anchors");

    DALI_ENFORCE((anchors.size() % 4) == 0);

    anchors_.Resize({(Index)anchors.size() / 4, 4});
    float *anchors_data = anchors_.template mutable_data<float>();

    MemCopy(anchors_data, anchors.data(), anchors.size() * sizeof(float));
  }

  virtual inline ~SSDBoxEncoder() = default;

  DISABLE_COPY_MOVE_ASSIGN(SSDBoxEncoder);

  const float criteria_;
  Tensor<Backend> anchors_;

 protected:
  void RunImpl(Workspace<Backend> *ws, const int idx) override;
};

}  // namespace dali

#endif  // DALI_PIPELINE_OPERATORS_DETECTION_BOX_ENCODER_H_
